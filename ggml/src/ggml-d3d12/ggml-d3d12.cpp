// ggml Direct3D 12 backend
//
// Design (see docs/build.md, "D3D12"):
//  - every tensor lives in a DEFAULT-heap buffer bound through root UAV descriptors (no descriptor heaps)
//  - kernel parameters go through a persistently mapped UPLOAD-heap arena bound as a root CBV
//  - HLSL is embedded as source and compiled at runtime with dxcompiler.dll (+ dxil.dll for signing)
//  - execution is synchronous: one direct queue, one command list, fence wait after every submit

#include "ggml-d3d12.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-d3d12-shaders.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef GGML_D3D12_DEBUG
#define D3D12_LOG_DEBUG(...) GGML_LOG_DEBUG("ggml_d3d12: " __VA_ARGS__)
#else
#define D3D12_LOG_DEBUG(...)
#endif

#define D3D12_WG_SIZE               256
#define D3D12_MAX_WG_PER_DIM        65535
#define D3D12_MAX_ROOT_UAVS         12   // 2 DWORDs each in the root signature
#define D3D12_BINDING_ALIGNMENT     256   // root CBV alignment, also used for tensor UAV base addresses
#define D3D12_PARAM_SLOT_SIZE       256
#define D3D12_PARAM_SLOT_COUNT      8192
#define D3D12_QUERY_CAPACITY        (2 * D3D12_PARAM_SLOT_COUNT)
#define D3D12_STAGING_SIZE          (64ull * 1024 * 1024)
#define D3D12_MEMSET_BYTES_PER_THREAD 16
#define D3D12_DEFAULT_MAX_ALLOC     (2ull * 1024 * 1024 * 1024)
#define D3D12_DEFAULT_SUBMIT_BATCH  64    // dispatches per command list (GGML_D3D12_SUBMIT_BATCH)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
// work budget per submission for kernels with long per-thread loops (flash attention, gated delta net,
// argsort): small enough that one command list stays well inside the Windows GPU timeout on slow cards
#define D3D12_FLASH_ATTN_WORK (1ull << 25)
#define D3D12_FLASH_ATTN_BLK      32                    // KV entries per flash attention block thread
#define D3D12_FLASH_ATTN_TMP_MAX  (64ull * 1024 * 1024)   // cap on the block results buffer

/* Minimal COM smart pointer (avoids a WRL dependency for MinGW builds) */

template <typename T> struct com_ptr {
    T * p = nullptr;

    com_ptr() = default;
    com_ptr(const com_ptr & o) : p(o.p) { if (p) { p->AddRef(); } }
    com_ptr(com_ptr && o) noexcept : p(o.p) { o.p = nullptr; }
    com_ptr & operator=(const com_ptr & o) {
        if (this != &o) { reset(); p = o.p; if (p) { p->AddRef(); } }
        return *this;
    }
    com_ptr & operator=(com_ptr && o) noexcept {
        if (this != &o) { reset(); p = o.p; o.p = nullptr; }
        return *this;
    }
    ~com_ptr() { reset(); }

    void reset() { if (p) { p->Release(); p = nullptr; } }
    T ** put() { reset(); return &p; }
    T * get() const { return p; }
    T * operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

static double ggml_d3d12_time_us() {
    static LARGE_INTEGER freq = {};
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double) now.QuadPart * 1e6 / (double) freq.QuadPart;
}

static void ggml_d3d12_check(HRESULT hr, const char * what) {
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d12: %s failed with HRESULT 0x%08lx\n", what, (unsigned long) hr);
        GGML_ABORT("ggml_d3d12: %s failed", what);
    }
}

static std::string ggml_d3d12_wide_to_utf8(const wchar_t * w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) {
        WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    }
    return s;
}

static std::wstring ggml_d3d12_utf8_to_wide(const std::string & s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) {
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    }
    return w;
}

// All device buffers report the same fake host base pointer; a tensor's byte offset inside its
// buffer is recovered from tensor->data (same scheme as the WebGPU backend).
static void * const d3d12_ptr_base = (void *) (uintptr_t) 0x1000;  // NOLINT

static size_t ggml_d3d12_tensor_offset(const ggml_tensor * tensor) {
    const ggml_tensor * base_tensor = tensor->view_src ? tensor->view_src : tensor;
    return (size_t) ((uintptr_t) base_tensor->data - (uintptr_t) d3d12_ptr_base) + tensor->view_offs;
}

/* Structs */

struct d3d12_caps {
    D3D_SHADER_MODEL shader_model = D3D_SHADER_MODEL_6_0;
    bool             native_16bit = false;
    bool             wave_ops     = false;
    uint32_t         wave_min     = 0;
    uint32_t         wave_max     = 0;
    bool             uma          = false;
};

struct d3d12_pipeline {
    com_ptr<ID3D12PipelineState> pso;
    std::string                  name;
};

struct d3d12_device_ctx {
    std::string name;   // "D3D120"
    std::string desc;   // adapter description
    size_t      dedicated_mem = 0;
    size_t      shared_mem    = 0;
    size_t      max_alloc     = D3D12_DEFAULT_MAX_ALLOC;
    d3d12_caps  caps;

    com_ptr<IDXGIAdapter1>            adapter;
    com_ptr<ID3D12Device>             device;
    com_ptr<ID3D12CommandQueue>       queue;
    com_ptr<ID3D12CommandAllocator>   allocator;
    com_ptr<ID3D12GraphicsCommandList> cmd_list;
    com_ptr<ID3D12Fence>              fence;
    uint64_t                          fence_value = 0;
    HANDLE                            fence_event = nullptr;
    com_ptr<ID3D12RootSignature>      root_sig;

    // staging for set/get tensor
    com_ptr<ID3D12Resource> upload_buf;
    void *                  upload_ptr = nullptr;
    com_ptr<ID3D12Resource> readback_buf;

    // flash attention block results, grown on demand
    com_ptr<ID3D12Resource> fa_tmp;
    size_t                  fa_tmp_size = 0;
    uint64_t                fa_work     = D3D12_FLASH_ATTN_WORK;   // adapted to the measured submit time

    // parameter arena (root CBV slots)
    com_ptr<ID3D12Resource>   param_buf;
    uint8_t *                 param_ptr = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS param_va  = 0;
    uint32_t                  param_next_slot = 0;
    uint32_t                  submit_batch    = D3D12_DEFAULT_SUBMIT_BATCH;
    uint32_t                  dispatches_in_list = 0;

    // shader compiler
    HMODULE                 dxc_module = nullptr;
    com_ptr<IDxcCompiler3>  compiler;
    com_ptr<IDxcUtils>      utils;
    std::unordered_map<std::string, d3d12_pipeline> pipelines;

    // parallel compilation: a collect pass over a new graph queues the missing pipelines, worker threads build them
    struct pipeline_job {
        std::string              key;
        const char *             source;
        std::vector<std::string> defines;
        D3D_SHADER_MODEL         min_sm;
    };
    bool                      collecting       = false;
    std::vector<pipeline_job> pipeline_jobs;
    std::mutex                pipelines_mutex;   // guards pipelines and the compile counters while workers run
    int                       last_graph_nodes = -1;

    bool recording = false;

    // counters printed at backend free when GGML_D3D12_STATS is set
    bool     stats            = false;
    uint64_t n_graphs         = 0;
    uint64_t n_nodes          = 0;
    uint64_t n_dispatches     = 0;
    uint64_t n_submits        = 0;
    uint64_t n_set_tensor     = 0;
    uint64_t n_get_tensor     = 0;
    uint64_t bytes_set        = 0;
    uint64_t bytes_get        = 0;
    double   t_wait_us        = 0;   // time inside fence waits
    double   t_submit_us      = 0;   // time inside Close + ExecuteCommandLists + Signal
    double   t_compile_us     = 0;   // time inside shader compilation (part of encode)
    uint64_t n_compiles       = 0;
    bool     no_barrier       = false; // GGML_D3D12_NO_BARRIER: measurement only, results are wrong
    bool     no_fuse          = false; // GGML_D3D12_NO_FUSE: encode every node on its own, for bisecting
    uint32_t mm_tpr_max       = D3D12_WG_SIZE; // GGML_D3D12_MM_TPR: cap on matvec threads per row (1 = no reduction tree)
    uint32_t tiled_min_cols   = 0;   // GGML_D3D12_TILED: columns from which the tiled prompt kernel is used (0 = never)
    std::string disable_ops;           // GGML_D3D12_DISABLE_OPS: comma separated op names sent to the CPU
    std::mutex  rejected_mutex;
    std::map<std::string, uint64_t> rejected;   // with stats: "op src types -> type" refused by supports_op
    uint64_t n_flush_arena    = 0;   // mid-graph submits because the param arena was full
    uint64_t n_flush_batch    = 0;   // mid-graph submits because submit_batch was reached
    double   t_graph_us       = 0;   // time inside graph_compute
    double   t_set_us         = 0;
    double   t_get_us         = 0;

    // GPU timestamp profiling per dispatch when GGML_D3D12_PROFILE is set
    bool                        profile = false;
    com_ptr<ID3D12QueryHeap>    query_heap;
    com_ptr<ID3D12Resource>     query_readback;
    uint64_t                    timestamp_freq = 0;
    uint32_t                    query_count    = 0;   // dispatches with queries in the open list
    std::vector<std::string>    query_names;
    std::map<std::string, std::pair<double, uint64_t>> prof;   // name -> (gpu us, count)

    std::recursive_mutex mutex;

    ggml_backend_buffer_type buft = {};

    ~d3d12_device_ctx() {
        if (fence_event) {
            CloseHandle(fence_event);
        }
    }
};

struct ggml_backend_d3d12_buffer_context {
    std::shared_ptr<d3d12_device_ctx> dev;
    com_ptr<ID3D12Resource>           res;
    D3D12_GPU_VIRTUAL_ADDRESS         va = 0;
    size_t                            size = 0;
};

struct ggml_backend_d3d12_context {
    std::shared_ptr<d3d12_device_ctx> dev;
    std::string                       name;
};

// the registry owns the device contexts; buffers and backends hold shared references
static std::shared_ptr<d3d12_device_ctx> ggml_d3d12_shared_dev(d3d12_device_ctx * dev);
static void ggml_d3d12_atexit();

/* Device helpers */

static com_ptr<ID3D12Resource> ggml_d3d12_create_buffer(d3d12_device_ctx & dev,
                                                        size_t             size,
                                                        D3D12_HEAP_TYPE    heap_type,
                                                        const wchar_t *    name) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type                  = heap_type;
    heap.CPUPageProperty       = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap.MemoryPoolPreference  = D3D12_MEMORY_POOL_UNKNOWN;
    heap.CreationNodeMask      = 1;
    heap.VisibleNodeMask       = 1;

    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension           = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment           = 0;
    desc.Width               = size;
    desc.Height              = 1;
    desc.DepthOrArraySize    = 1;
    desc.MipLevels           = 1;
    desc.Format              = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count    = 1;
    desc.SampleDesc.Quality  = 0;
    desc.Layout              = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags               = heap_type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                                                                    : D3D12_RESOURCE_FLAG_NONE;

    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    if (heap_type == D3D12_HEAP_TYPE_UPLOAD) {
        state = D3D12_RESOURCE_STATE_GENERIC_READ;
    } else if (heap_type == D3D12_HEAP_TYPE_READBACK) {
        state = D3D12_RESOURCE_STATE_COPY_DEST;
    }

    com_ptr<ID3D12Resource> res;
    HRESULT hr = dev.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(res.put()));
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d12: failed to allocate %zu bytes (heap type %d), HRESULT 0x%08lx\n", size,
                       (int) heap_type, (unsigned long) hr);
        return {};
    }
    if (name) {
        res->SetName(name);
    }
    return res;
}

// begin recording on the (single) command list; the previous submission has always completed
static void ggml_d3d12_begin(d3d12_device_ctx & dev, bool compute) {
    GGML_ASSERT(!dev.recording);
    ggml_d3d12_check(dev.allocator->Reset(), "ID3D12CommandAllocator::Reset");
    ggml_d3d12_check(dev.cmd_list->Reset(dev.allocator.get(), nullptr), "ID3D12GraphicsCommandList::Reset");
    if (compute) {
        dev.cmd_list->SetComputeRootSignature(dev.root_sig.get());
    }
    dev.recording = true;
}

static void ggml_d3d12_submit_and_wait(d3d12_device_ctx & dev) {
    GGML_ASSERT(dev.recording);
    if (dev.query_count > 0) {
        dev.cmd_list->ResolveQueryData(dev.query_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 * dev.query_count,
                                       dev.query_readback.get(), 0);
    }
    const double ts = ggml_d3d12_time_us();
    ggml_d3d12_check(dev.cmd_list->Close(), "ID3D12GraphicsCommandList::Close");
    ID3D12CommandList * lists[] = { dev.cmd_list.get() };
    dev.queue->ExecuteCommandLists(1, lists);
    const uint64_t v = ++dev.fence_value;
    ggml_d3d12_check(dev.queue->Signal(dev.fence.get(), v), "ID3D12CommandQueue::Signal");
    const double t0 = ggml_d3d12_time_us();
    dev.t_submit_us += t0 - ts;
    if (dev.fence->GetCompletedValue() < v) {
        ggml_d3d12_check(dev.fence->SetEventOnCompletion(v, dev.fence_event), "ID3D12Fence::SetEventOnCompletion");
        WaitForSingleObject(dev.fence_event, INFINITE);
    }
    dev.t_wait_us += ggml_d3d12_time_us() - t0;
    dev.n_submits++;
    dev.recording           = false;
    dev.param_next_slot     = 0;
    dev.dispatches_in_list  = 0;

    if (dev.query_count > 0) {
        void *      ptr   = nullptr;
        D3D12_RANGE range = { 0, 2 * dev.query_count * sizeof(uint64_t) };
        if (SUCCEEDED(dev.query_readback->Map(0, &range, &ptr))) {
            const uint64_t * ts = (const uint64_t *) ptr;
            for (uint32_t i = 0; i < dev.query_count; i++) {
                const double us = (double) (ts[2 * i + 1] - ts[2 * i]) * 1e6 / (double) dev.timestamp_freq;
                auto & e = dev.prof[dev.query_names[i]];
                e.first += us;
                e.second++;
            }
            D3D12_RANGE no_write = { 0, 0 };
            dev.query_readback->Unmap(0, &no_write);
        }
        dev.query_count = 0;
        dev.query_names.clear();
    }

    HRESULT removed = dev.device->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        GGML_LOG_ERROR("ggml_d3d12: device removed, reason 0x%08lx\n", (unsigned long) removed);
        GGML_ABORT("ggml_d3d12: device removed");
    }
}

/* Shader compilation */

static const char * ggml_d3d12_profile(D3D_SHADER_MODEL sm) {
    switch (sm) {
        case D3D_SHADER_MODEL_6_0: return "cs_6_0";
        case D3D_SHADER_MODEL_6_1: return "cs_6_1";
        case D3D_SHADER_MODEL_6_2: return "cs_6_2";
        case D3D_SHADER_MODEL_6_3: return "cs_6_3";
        case D3D_SHADER_MODEL_6_4: return "cs_6_4";
        case D3D_SHADER_MODEL_6_5: return "cs_6_5";
        case D3D_SHADER_MODEL_6_6: return "cs_6_6";
        default:                   return "cs_6_7";
    }
}

// compiled DXIL is cached in d3d12-shader-cache next to ggml-d3d12.dll, named by a hash of the source and
// the compiler arguments; GGML_D3D12_NO_SHADER_CACHE disables it. Unwritable folders just skip the cache.
static std::wstring ggml_d3d12_shader_cache_file(const char * source, const std::vector<std::wstring> & args) {
    static const bool disabled = getenv("GGML_D3D12_NO_SHADER_CACHE") != nullptr;
    if (disabled) {
        return L"";
    }
    uint64_t h = 0xcbf29ce484222325ULL;
    auto hash_bytes = [&h](const void * data, size_t size) {
        const unsigned char * b = (const unsigned char *) data;
        for (size_t i = 0; i < size; i++) {
            h = (h ^ b[i]) * 0x100000001b3ULL;
        }
    };
    hash_bytes(source, strlen(source));
    for (const auto & a : args) {
        hash_bytes(a.c_str(), (a.size() + 1) * sizeof(wchar_t));
    }
    HMODULE module = nullptr;
    wchar_t path[MAX_PATH];
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR) &ggml_d3d12_shader_cache_file, &module) ||
        GetModuleFileNameW(module, path, MAX_PATH) - 1 >= MAX_PATH - 1) {   // 0 = error, MAX_PATH = truncated
        return L"";
    }
    std::wstring dir = path;
    dir = dir.substr(0, dir.find_last_of(L"\\/") + 1) + L"d3d12-shader-cache";
    CreateDirectoryW(dir.c_str(), nullptr);
    std::wstring name = L"\\0000000000000000.dxil";
    for (int i = 16; i > 0; i--, h >>= 4) {
        name[i] = L"0123456789abcdef"[h & 0xf];
    }
    return dir + name;
}

static std::vector<uint8_t> ggml_d3d12_read_file(const std::wstring & file) {
    std::vector<uint8_t> data;
    FILE * f = file.empty() ? nullptr : _wfopen(file.c_str(), L"rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size > 0) {
            data.resize((size_t) size);
            if (fread(data.data(), 1, data.size(), f) != data.size()) {
                data.clear();
            }
        }
        fclose(f);
    }
    // a DXIL container starts with DXBC
    if (data.size() < 32 || memcmp(data.data(), "DXBC", 4) != 0) {
        data.clear();
    }
    return data;
}

// write to a per-process temp name and rename, so concurrent processes never see a partial file
static void ggml_d3d12_write_file(const std::wstring & file, const void * data, size_t size) {
    if (file.empty()) {
        return;
    }
    const std::wstring tmp = file + L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
    FILE * f = _wfopen(tmp.c_str(), L"wb");
    if (!f) {
        return;
    }
    const bool ok = fwrite(data, 1, size, f) == size;
    fclose(f);
    if (!ok || !MoveFileExW(tmp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(tmp.c_str());
    }
}

// builds one pipeline from the disk cache or by compiling; thread safe when each thread passes its own compiler
static d3d12_pipeline ggml_d3d12_build_pipeline(d3d12_device_ctx &               dev,
                                               IDxcCompiler3 *                  compiler,
                                               const std::string &              key,
                                               const char *                     source,
                                               const std::vector<std::string> & defines,
                                               D3D_SHADER_MODEL                 min_sm) {
    const double t_compile0 = ggml_d3d12_time_us();

    const bool use_16bit = std::find(defines.begin(), defines.end(), "USE_16BIT") != defines.end();
    D3D_SHADER_MODEL sm  = std::max(min_sm, use_16bit ? D3D_SHADER_MODEL_6_2 : D3D_SHADER_MODEL_6_0);
    GGML_ASSERT(sm <= dev.caps.shader_model);

    std::vector<std::wstring> args = { L"-E", L"main", L"-T", ggml_d3d12_utf8_to_wide(ggml_d3d12_profile(sm)),
                                       L"-O3", L"-DWG_SIZE=" + std::to_wstring(D3D12_WG_SIZE) };
    if (use_16bit) {
        args.push_back(L"-enable-16bit-types");
    }
#ifdef GGML_D3D12_DEBUG
    args.push_back(L"-Zi");
    args.push_back(L"-Qembed_debug");
#endif
    for (const auto & d : defines) {
        args.push_back(L"-D" + ggml_d3d12_utf8_to_wide(d));
    }
    std::vector<LPCWSTR> argv;
    for (const auto & a : args) {
        argv.push_back(a.c_str());
    }

    const std::wstring   cache_file = ggml_d3d12_shader_cache_file(source, args);
    for (int attempt = 0;; attempt++) {
        std::vector<uint8_t> dxil   = attempt == 0 ? ggml_d3d12_read_file(cache_file) : std::vector<uint8_t>();
        const bool           cached = !dxil.empty();
        if (!cached) {
            DxcBuffer src = {};
            src.Ptr       = source;
            src.Size      = strlen(source);
            src.Encoding  = DXC_CP_UTF8;

            com_ptr<IDxcResult> result;
            ggml_d3d12_check(compiler->Compile(&src, argv.data(), (UINT32) argv.size(), nullptr, IID_PPV_ARGS(result.put())),
                             "IDxcCompiler3::Compile");

            com_ptr<IDxcBlobUtf8> errors;
            result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errors.put()), nullptr);
            HRESULT status = S_OK;
            result->GetStatus(&status);
            if (FAILED(status)) {
                GGML_LOG_ERROR("ggml_d3d12: shader compilation failed for %s:\n%s\n", key.c_str(),
                               errors && errors->GetStringLength() ? errors->GetStringPointer() : "(no output)");
                GGML_ABORT("ggml_d3d12: shader compilation failed");
            }
#ifdef GGML_D3D12_DEBUG
            if (errors && errors->GetStringLength()) {
                GGML_LOG_WARN("ggml_d3d12: shader %s warnings:\n%s\n", key.c_str(), errors->GetStringPointer());
            }
#endif

            com_ptr<IDxcBlob> object;
            ggml_d3d12_check(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(object.put()), nullptr), "IDxcResult::GetOutput");
            const uint8_t * bytes = (const uint8_t *) object->GetBufferPointer();
            dxil.assign(bytes, bytes + object->GetBufferSize());
            ggml_d3d12_write_file(cache_file, dxil.data(), dxil.size());
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
        pso_desc.pRootSignature                    = dev.root_sig.get();
        pso_desc.CS.pShaderBytecode                = dxil.data();
        pso_desc.CS.BytecodeLength                 = dxil.size();

        d3d12_pipeline pipeline;
        pipeline.name = key;
        HRESULT hr    = dev.device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(pipeline.pso.put()));
        if (FAILED(hr) && cached) {
            // a stale or damaged cache entry: drop it and compile from source
            GGML_LOG_WARN("ggml_d3d12: cached shader rejected for %s, recompiling\n", key.c_str());
            DeleteFileW(cache_file.c_str());
            continue;
        }
        if (FAILED(hr)) {
            GGML_LOG_ERROR("ggml_d3d12: CreateComputePipelineState failed for %s (HRESULT 0x%08lx). "
                           "If dxil.dll is missing next to dxcompiler.dll the shader is unsigned and rejected.\n",
                           key.c_str(), (unsigned long) hr);
            GGML_ABORT("ggml_d3d12: pipeline creation failed");
        }
        D3D12_LOG_DEBUG("compiled pipeline %s\n", key.c_str());
        std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
        dev.n_compiles++;
        dev.t_compile_us += ggml_d3d12_time_us() - t_compile0;
        return pipeline;
    }
}

// defines: list of "NAME" or "NAME=VALUE"
static d3d12_pipeline & ggml_d3d12_get_pipeline(d3d12_device_ctx &               dev,
                                                const char *                     shader_name,
                                                const char *                     source,
                                                const std::vector<std::string> & defines,
                                                D3D_SHADER_MODEL                 min_sm = D3D_SHADER_MODEL_6_0) {
    std::string key = shader_name;
    for (const auto & d : defines) {
        key += " -D" + d;
    }
    if (min_sm > D3D_SHADER_MODEL_6_0) {
        key += " sm" + std::to_string((int) min_sm);
    }

    auto it = dev.pipelines.find(key);
    if (it != dev.pipelines.end()) {
        return it->second;
    }
    if (dev.collecting) {
        // collect pass: queue the pipeline once; the pass records no dispatches, so a placeholder is enough
        static d3d12_pipeline placeholder;
        const bool queued = std::any_of(dev.pipeline_jobs.begin(), dev.pipeline_jobs.end(),
                                        [&](const d3d12_device_ctx::pipeline_job & j) { return j.key == key; });
        if (!queued) {
            dev.pipeline_jobs.push_back({ key, source, defines, min_sm });
        }
        return placeholder;
    }
    d3d12_pipeline pipeline = ggml_d3d12_build_pipeline(dev, dev.compiler.get(), key, source, defines, min_sm);
    return dev.pipelines.emplace(key, std::move(pipeline)).first->second;
}

// builds the queued pipelines on worker threads, each with its own compiler instance
static void ggml_d3d12_build_pipeline_jobs(d3d12_device_ctx & dev) {
    std::vector<d3d12_device_ctx::pipeline_job> jobs;
    jobs.swap(dev.pipeline_jobs);
    if (jobs.empty()) {
        return;
    }
    auto           create    = (DxcCreateInstanceProc) (void *) GetProcAddress(dev.dxc_module, "DxcCreateInstance");
    const size_t   n_threads = std::min<size_t>(jobs.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<size_t> next{ 0 };
    auto worker = [&]() {
        com_ptr<IDxcCompiler3> compiler;
        ggml_d3d12_check(create(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.put())), "DxcCreateInstance");
        for (size_t i; (i = next++) < jobs.size();) {
            const auto &   j = jobs[i];
            d3d12_pipeline p = ggml_d3d12_build_pipeline(dev, compiler.get(), j.key, j.source, j.defines, j.min_sm);
            std::lock_guard<std::mutex> lock(dev.pipelines_mutex);
            dev.pipelines.emplace(j.key, std::move(p));
        }
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < n_threads; t++) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & th : threads) {
        th.join();
    }
}

/* Dispatch encoding */

static inline void ggml_d3d12_workgroups_2d(uint32_t total_wg, uint32_t & wg_x, uint32_t & wg_y) {
    wg_y = std::max(1u, CEIL_DIV(total_wg, (uint32_t) D3D12_MAX_WG_PER_DIM));
    wg_x = CEIL_DIV(total_wg, wg_y);
}

static inline uint32_t ggml_d3d12_u32_from_f32(float value) {
    uint32_t u;
    memcpy(&u, &value, sizeof(u));
    return u;
}

struct d3d12_binding {
    D3D12_GPU_VIRTUAL_ADDRESS va;
    uint32_t                  elem_offset;   // misalignment in elements, passed to the kernel
};

static D3D12_GPU_VIRTUAL_ADDRESS ggml_d3d12_tensor_va(const ggml_tensor * t) {
    auto * buf_ctx = (ggml_backend_d3d12_buffer_context *) t->buffer->context;
    return buf_ctx->va;
}

// Bind at the largest 256-byte aligned offset at or before the tensor such that the distance is a
// whole number of type blocks, so the kernel can index the remainder in elements.
static d3d12_binding ggml_d3d12_bind_tensor(const ggml_tensor * t) {
    const size_t offset    = ggml_d3d12_tensor_offset(t);
    const size_t type_size = ggml_type_size(t->type);
    size_t       aligned   = offset & ~((size_t) D3D12_BINDING_ALIGNMENT - 1);
    while ((offset - aligned) % type_size != 0) {
        GGML_ASSERT(aligned >= D3D12_BINDING_ALIGNMENT);
        aligned -= D3D12_BINDING_ALIGNMENT;
    }
    d3d12_binding b;
    b.va          = ggml_d3d12_tensor_va(t) + aligned;
    b.elem_offset = (uint32_t) ((offset - aligned) / type_size);
    return b;
}

// records one dispatch on the open compute command list (params get nwg_x appended)
static void ggml_d3d12_dispatch(d3d12_device_ctx &                       dev,
                                d3d12_pipeline &                         pipeline,
                                std::vector<uint32_t>                    params,
                                const std::vector<D3D12_GPU_VIRTUAL_ADDRESS> & uavs,
                                uint32_t                                 total_wg) {
    GGML_ASSERT(dev.recording);
    GGML_ASSERT(uavs.size() <= D3D12_MAX_ROOT_UAVS);
    if (dev.collecting) {
        return;
    }

    uint32_t wg_x, wg_y;
    ggml_d3d12_workgroups_2d(total_wg, wg_x, wg_y);
    params.push_back(wg_x);
    GGML_ASSERT(params.size() * sizeof(uint32_t) <= D3D12_PARAM_SLOT_SIZE);

    if (dev.param_next_slot >= D3D12_PARAM_SLOT_COUNT || dev.dispatches_in_list >= dev.submit_batch) {
        // arena exhausted or batch full: flush what we have and start a new list
        if (dev.param_next_slot >= D3D12_PARAM_SLOT_COUNT) { dev.n_flush_arena++; } else { dev.n_flush_batch++; }
        ggml_d3d12_submit_and_wait(dev);
        ggml_d3d12_begin(dev, true);
    }
    dev.dispatches_in_list++;
    const uint32_t slot = dev.param_next_slot++;
    memcpy(dev.param_ptr + (size_t) slot * D3D12_PARAM_SLOT_SIZE, params.data(), params.size() * sizeof(uint32_t));

    dev.cmd_list->SetPipelineState(pipeline.pso.get());
    dev.cmd_list->SetComputeRootConstantBufferView(0, dev.param_va + (uint64_t) slot * D3D12_PARAM_SLOT_SIZE);
    for (size_t i = 0; i < uavs.size(); i++) {
        dev.cmd_list->SetComputeRootUnorderedAccessView((UINT) (1 + i), uavs[i]);
    }
    const bool timed = dev.profile && dev.query_count < D3D12_QUERY_CAPACITY / 2;
    if (timed) {
        dev.cmd_list->EndQuery(dev.query_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 * dev.query_count);
    }
    dev.cmd_list->Dispatch(wg_x, wg_y, 1);
    dev.n_dispatches++;

    // TODO: only barrier between dependent nodes
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource          = nullptr;
    if (!dev.no_barrier) {
        dev.cmd_list->ResourceBarrier(1, &barrier);
    }

    // the end stamp goes after the barrier: before it, the stamp only measured dispatch issue
    if (timed) {
        dev.cmd_list->EndQuery(dev.query_heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 * dev.query_count + 1);
        dev.query_names.push_back(pipeline.name);
        dev.query_count++;
    }
}

// memset [offset, offset+size) bytes of a buffer with a replicated byte value; standalone submission
static void ggml_d3d12_buffer_memset(d3d12_device_ctx & dev, D3D12_GPU_VIRTUAL_ADDRESS va, size_t offset, size_t size, uint8_t value) {
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    const uint32_t val32 = (uint32_t) value * 0x01010101u;

    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "memset", hlsl_memset,
                                                        { "BYTES_PER_THREAD=" + std::to_string(D3D12_MEMSET_BYTES_PER_THREAD) });
    ggml_d3d12_begin(dev, true);
    // chunk so that offsets inside a chunk fit in 32 bits; chunks start at 256-byte aligned addresses
    const size_t chunk_max = 1ull << 30;
    size_t done = 0;
    while (done < size) {
        const size_t   abs_off   = offset + done;
        const size_t   base      = abs_off & ~((size_t) D3D12_BINDING_ALIGNMENT - 1);
        const uint32_t rel_off   = (uint32_t) (abs_off - base);
        const uint32_t n         = (uint32_t) std::min(size - done, chunk_max);
        const uint32_t span      = rel_off + n;   // bytes covered from `base`
        const uint32_t threads   = CEIL_DIV(span, (uint32_t) D3D12_MEMSET_BYTES_PER_THREAD);
        ggml_d3d12_dispatch(dev, pipeline, { rel_off, n, val32 }, { va + base }, CEIL_DIV(threads, (uint32_t) D3D12_WG_SIZE));
        done += n;
    }
    ggml_d3d12_submit_and_wait(dev);
}

/* Op encoders */

static std::string ggml_d3d12_type_define(ggml_type type, const char * prefix) {
    std::string s = prefix;
    switch (type) {
        case GGML_TYPE_F32: s += "_F32"; break;
        case GGML_TYPE_F16: s += "_F16"; break;
        case GGML_TYPE_I32: s += "_I32"; break;
        default: GGML_ABORT("ggml_d3d12: unsupported type %s", ggml_type_name(type));
    }
    return s;
}

static void ggml_d3d12_cpy(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines = { ggml_d3d12_type_define(src->type, "SRC"), ggml_d3d12_type_define(dst->type, "DST") };
    if (src->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "cpy", hlsl_cpy, defines);

    const d3d12_binding bsrc = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bdst = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne   = (uint32_t) ggml_nelements(dst);
    const size_t        ts   = ggml_type_size(src->type);
    const size_t        td   = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        ne, bsrc.elem_offset, bdst.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[0] / td), (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bsrc.va, bdst.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_binary_op(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    const char * op_define = nullptr;
    switch (dst->op) {
        case GGML_OP_ADD: op_define = "OP_ADD"; break;
        case GGML_OP_SUB: op_define = "OP_SUB"; break;
        case GGML_OP_MUL: op_define = "OP_MUL"; break;
        case GGML_OP_DIV: op_define = "OP_DIV"; break;
        default: GGML_ABORT("ggml_d3d12: unexpected binary op");
    }
    std::vector<std::string> defines = { ggml_d3d12_type_define(dst->type, "TYPE"), op_define };
    if (dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "binary", hlsl_binary, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const size_t        t0 = ggml_type_size(src0->type);
    const size_t        t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        ne, b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / t0), (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[0] / t1), (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) src1->ne[0], (uint32_t) src1->ne[1], (uint32_t) src1->ne[2], (uint32_t) src1->ne[3],
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_scale(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "scale", hlsl_scale, {});

    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);

    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        ne, (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 0)),   // scale
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 1)),   // bias
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

// SET_ROWS into q8_0 (quantized KV cache): two dispatches, rows of even and odd dst row numbers
static void ggml_d3d12_set_rows_q8_0(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (idx->type == GGML_TYPE_I64) {
        defines.push_back("I64_IDX");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "set_rows_q8", hlsl_set_rows_q8, defines);

    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(idx);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        ti = ggml_type_size(idx->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      n_rows = (uint32_t) (src->ne[1] * src->ne[2] * src->ne[3]);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / ti), (uint32_t) (idx->nb[1] / ti), (uint32_t) (idx->nb[2] / ti),
        (uint32_t) (dst->nb[1] / td), (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) idx->ne[1], (uint32_t) idx->ne[2], 0,
    };
    for (uint32_t parity = 0; parity < 2; parity++) {
        params.back() = parity;
        ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_rows, (uint32_t) D3D12_WG_SIZE));
    }
}

static void ggml_d3d12_set_rows(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_empty(src) || ggml_is_empty(idx)) {
        return;
    }
    if (dst->type == GGML_TYPE_Q8_0) {
        ggml_d3d12_set_rows_q8_0(dev, src, idx, dst);
        return;
    }
    std::vector<std::string> defines = { ggml_d3d12_type_define(dst->type, "DST") };
    if (dst->type == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    if (idx->type == GGML_TYPE_I64) {
        defines.push_back("I64_IDX");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "set_rows", hlsl_set_rows, defines);

    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(idx);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        ti = ggml_type_size(idx->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(src);

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / ti), (uint32_t) (idx->nb[1] / ti), (uint32_t) (idx->nb[2] / ti),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) idx->ne[1], (uint32_t) idx->ne[2],
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static bool ggml_d3d12_mul_mat_vec_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

// one matrix of a fused mul_mat_vec dispatch: dst = src0 * src1 (+ add, broadcast like ggml_add)
struct d3d12_mat_slot {
    ggml_tensor * src0;
    ggml_tensor * dst;   // the MUL_MAT node, or the fused ADD node
    ggml_tensor * add;   // addend of the fused ADD, or null
};

// types the tiled prompt kernel dequantizes (mul_mat_tiled.hlsl)
static bool ggml_d3d12_tiled_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// Is this product worth the tiled kernel, and can that kernel express it? Long prompts only: for a
// handful of columns the matvec kernel wins, because a tile of 32 columns would be mostly padding.
static bool ggml_d3d12_use_tiled(const d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                 ggml_tensor * dst) {
    return dev.tiled_min_cols != 0 && (uint32_t) dst->ne[1] >= dev.tiled_min_cols &&
           ggml_d3d12_tiled_type(src0->type) && src0->ne[0] % 32 == 0 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && dst->type == GGML_TYPE_F32;
}

// dst = src0 * src1 with a TILE_M x TILE_N tile of dst per workgroup; see mul_mat_tiled.hlsl
static void ggml_d3d12_mul_mat_tiled(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1,
                                     ggml_tensor * dst) {
    std::string define = "SRC0_";
    define += ggml_type_name(src0->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    std::vector<std::string> defines = { define };
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "mul_mat_tiled", hlsl_mul_mat_tiled, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        t0 = ggml_type_size(src0->type);
    const size_t        t1 = ggml_type_size(src1->type);

    const uint32_t broadcast2 = (uint32_t) (src1->ne[2] / src0->ne[2]);
    const uint32_t broadcast3 = (uint32_t) (src1->ne[3] / src0->ne[3]);

    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src0->nb[1] / t0), (uint32_t) (src0->nb[2] / t0), (uint32_t) (src0->nb[3] / t0),
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) dst->ne[2], broadcast2, broadcast3,
        (uint32_t) (dst->ne[2] * dst->ne[3]),
    };

    // tile sizes must match TILE_M and TILE_N in mul_mat_tiled.hlsl
    const uint32_t tiles_m  = CEIL_DIV((uint32_t) dst->ne[0], 64u);
    const uint32_t tiles_n  = CEIL_DIV((uint32_t) dst->ne[1], 32u);
    const uint32_t batches  = (uint32_t) (dst->ne[2] * dst->ne[3]);
    ggml_d3d12_dispatch(dev, pipeline, params, { b1.va, b0.va, bd.va }, tiles_m * tiles_n * batches);
}

// up to 3 matrices sharing src1 in one dispatch; every matrix owns a range of workgroups
static void ggml_d3d12_mul_mat_group(d3d12_device_ctx & dev, ggml_tensor * src1, const std::vector<d3d12_mat_slot> & mats) {
    GGML_ASSERT(!mats.empty() && mats.size() <= 3);
    ggml_tensor * src0 = mats[0].src0;
    // a single unfused product with many columns goes to the tiled kernel instead
    if (mats.size() == 1 && mats[0].add == nullptr &&
        ggml_d3d12_use_tiled(dev, src0, src1, mats[0].dst)) {
        ggml_d3d12_mul_mat_tiled(dev, src0, src1, mats[0].dst);
        return;
    }
    std::string define = "SRC0_";
    define += ggml_type_name(src0->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    // threads per row: one unit of work (4 floats or one 32-wide block) per thread, power of two, at most WG_SIZE
    const uint32_t units = (uint32_t) (ggml_is_quantized(src0->type) ? src0->ne[0] / 32 : src0->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < D3D12_WG_SIZE && tpr < dev.mm_tpr_max) {
        tpr *= 2;
    }
    bool fuse_add = false;
    for (const auto & m : mats) {
        fuse_add = fuse_add || m.add != nullptr;
    }
    // single-column (decode) products skip the per-element column loop (+50% decode on the MTT S80).
    // Not for F32 weights: on the S80 that variant failed MUL_MAT f32 k=256 bs=[3,2] nr=[1,2] (ERR 0.3,
    // deterministic) while F16 and the quant types passed the same shape; cause not found yet.
    const bool     one_col  = mats[0].dst->ne[1] == 1 && src0->type != GGML_TYPE_F32;
    const uint32_t max_cols = one_col ? 1 : 4;
    std::vector<std::string> defines = { define, "TPR=" + std::to_string(tpr), "N_MATS=" + std::to_string(mats.size()) };
    if (one_col) {
        defines.push_back("ONE_COL");
    }
    if (fuse_add) {
        defines.push_back("FUSE_ADD");
    }
    if (src1->type == GGML_TYPE_F16) {
        defines.push_back("SRC1_F16");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "mul_mat_vec", hlsl_mul_mat_vec, defines);
    const uint32_t rows_per_wg = D3D12_WG_SIZE / tpr;

    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const size_t        t0 = ggml_type_size(src0->type);   // block size in bytes for quant types
    const size_t        t1 = ggml_type_size(src1->type);
    ggml_tensor * dst0 = mats[0].dst;

    std::vector<uint32_t> params = {
        b1.elem_offset, (uint32_t) dst0->ne[1], (uint32_t) src0->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        (uint32_t) src0->ne[2], (uint32_t) src0->ne[3],
        (uint32_t) (src1->ne[2] / src0->ne[2]), (uint32_t) (src1->ne[3] / src0->ne[3]),
        0,   // col0, set per chunk below
        0, 0 // wg_start_1, wg_start_2
    };
    const size_t col0_idx = 10;
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> uavs = { b1.va };
    uint32_t total_wg = 0;
    for (size_t i = 0; i < 3; i++) {
        if (i >= mats.size()) {
            for (int j = 0; j < 14; j++) {
                params.push_back(0);
            }
            continue;
        }
        const d3d12_mat_slot & m  = mats[i];
        const d3d12_binding    b0 = ggml_d3d12_bind_tensor(m.src0);
        const d3d12_binding    bd = ggml_d3d12_bind_tensor(m.dst);
        const d3d12_binding    ba = m.add ? ggml_d3d12_bind_tensor(m.add) : bd;
        if (i > 0) {
            params[10 + i] = total_wg;
        }
        params.insert(params.end(), {
            b0.elem_offset, bd.elem_offset, (uint32_t) m.dst->ne[0],
            (uint32_t) (m.src0->nb[1] / t0), (uint32_t) (m.src0->nb[2] / t0), (uint32_t) (m.src0->nb[3] / t0),
            m.add ? 1u : 0u, ba.elem_offset,
            m.add ? (uint32_t) m.add->ne[1] : 1u, m.add ? (uint32_t) m.add->ne[2] : 1u, m.add ? (uint32_t) m.add->ne[3] : 1u,
            m.add ? (uint32_t) (m.add->nb[1] / 4) : 0u, m.add ? (uint32_t) (m.add->nb[2] / 4) : 0u, m.add ? (uint32_t) (m.add->nb[3] / 4) : 0u,
        });
        uavs.insert(uavs.end(), { b0.va, bd.va, ba.va });
        total_wg += CEIL_DIV((uint32_t) m.dst->ne[0], rows_per_wg) * (uint32_t) (m.dst->ne[2] * m.dst->ne[3]);
    }
    // columns are processed 4 at a time; every chunk re-reads src0 (fine for decode, slow for long prompts)
    for (uint32_t col0 = 0; col0 < (uint32_t) dst0->ne[1]; col0 += max_cols) {
        params[col0_idx] = col0;
        ggml_d3d12_dispatch(dev, pipeline, params, uavs, total_wg);
    }
}

// MUL_MAT_ID: dst[:, slot, token] = as[:, :, ids[slot, token]]^T * src1[:, slot % ne11, token]
static void ggml_d3d12_mul_mat_id(d3d12_device_ctx & dev, ggml_tensor * as, ggml_tensor * src1, ggml_tensor * ids,
                                  ggml_tensor * dst) {
    std::string define = "SRC0_";
    define += ggml_type_name(as->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    const uint32_t units = (uint32_t) (ggml_is_quantized(as->type) ? as->ne[0] / 32 : as->ne[0] / 4);
    uint32_t       tpr   = 1;
    while (tpr < units && tpr < D3D12_WG_SIZE && tpr < dev.mm_tpr_max) {
        tpr *= 2;
    }
    std::vector<std::string> defines = { define, "TPR=" + std::to_string(tpr), "N_MATS=1", "MMID" };
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "mul_mat_vec", hlsl_mul_mat_vec, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(as);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(ids);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        t0 = ggml_type_size(as->type);
    const size_t        t1 = ggml_type_size(src1->type);

    std::vector<uint32_t> params = {
        b1.elem_offset, 1u, (uint32_t) as->ne[0],
        (uint32_t) (src1->nb[1] / t1), (uint32_t) (src1->nb[2] / t1), (uint32_t) (src1->nb[3] / t1),
        1u, 1u, 1u, 1u,
        0u,      // col0
        0u, 0u,  // wg_start_1, wg_start_2
    };
    // matrix slot 0 describes the expert matrices; slots 1 and 2 stay empty
    params.insert(params.end(), {
        b0.elem_offset, bd.elem_offset, (uint32_t) as->ne[1],
        (uint32_t) (as->nb[1] / t0), (uint32_t) (as->nb[2] / t0), (uint32_t) (as->nb[3] / t0),
        0u, bd.elem_offset, 1u, 1u, 1u, 0u, 0u, 0u,
    });
    for (int i = 0; i < 28; i++) {
        params.push_back(0);
    }
    params.insert(params.end(), {
        bi.elem_offset, (uint32_t) (ids->nb[1] / 4), (uint32_t) ids->ne[0], (uint32_t) src1->ne[1],
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
    });

    const uint32_t rows_per_wg = D3D12_WG_SIZE / tpr;
    const uint32_t total_wg    = CEIL_DIV((uint32_t) as->ne[1], rows_per_wg) *
                                 (uint32_t) (ids->ne[0] * ids->ne[1]);
    ggml_d3d12_dispatch(dev, pipeline, params, { b1.va, b0.va, bd.va, bd.va, bi.va }, total_wg);
}

static void ggml_d3d12_mul_mat(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    ggml_d3d12_mul_mat_group(dev, src1, { { src0, dst, nullptr } });
}

// can `add_node` (an ADD consuming `mm`) be folded into the matvec that produces `mm`? returns the addend
static ggml_tensor * ggml_d3d12_fusable_addend(const ggml_tensor * mm, const ggml_tensor * add_node) {
    if (add_node->src[0] != mm && add_node->src[1] != mm) {
        return nullptr;
    }
    ggml_tensor * other = add_node->src[0] == mm ? add_node->src[1] : add_node->src[0];
    if (other == mm || other->type != GGML_TYPE_F32 || other->nb[0] != sizeof(float) || other->ne[0] != mm->ne[0]) {
        return nullptr;
    }
    // ggml_add broadcasts src1 onto src0: when mm is src1, the shapes must match outright
    if (add_node->src[1] == mm && !ggml_are_same_shape(mm, other)) {
        return nullptr;
    }
    for (int d = 1; d < 4; d++) {
        if (other->ne[d] != 1 && other->ne[d] != mm->ne[d]) {
            return nullptr;
        }
    }
    if (add_node->type != GGML_TYPE_F32 || !ggml_is_contiguous(add_node)) {
        return nullptr;
    }
    return other;
}

// encode the MUL_MAT at node i together with up to two following MUL_MATs that share src1, each with its
// optional bias/residual ADD; returns the number of graph nodes consumed
static int ggml_d3d12_encode_mul_mat_group(d3d12_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * first = cgraph->nodes[i];
    ggml_tensor * src1  = first->src[1];
    std::vector<d3d12_mat_slot> mats;
    int j = i;
    const size_t max_mats = dev.no_fuse ? 1 : 3;
    while (j < cgraph->n_nodes && mats.size() < max_mats) {
        ggml_tensor * node = cgraph->nodes[j];
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != src1 || node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) ||
            node->src[0]->type != first->src[0]->type || node->src[0]->ne[0] != first->src[0]->ne[0] ||
            node->src[0]->ne[2] != first->src[0]->ne[2] || node->src[0]->ne[3] != first->src[0]->ne[3] ||
            node->src[0]->nb[0] != ggml_type_size(node->src[0]->type)) {
            break;
        }
        bool independent = true;
        for (const auto & m : mats) {
            independent = independent && node->src[0] != m.dst;
        }
        if (!independent) {
            break;
        }
        d3d12_mat_slot slot = { node->src[0], node, nullptr };
        int consumed = 1;
        if (!dev.no_fuse && ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
            ggml_tensor * addend = ggml_d3d12_fusable_addend(node, cgraph->nodes[j + 1]);
            bool ok = addend != nullptr;
            for (const auto & m : mats) {
                ok = ok && addend != m.dst && addend != m.src0;
            }
            if (ok) {
                slot.dst = cgraph->nodes[j + 1];
                slot.add = addend;
                consumed = 2;
            }
        }
        mats.push_back(slot);
        j += consumed;
    }
    if (mats.empty()) {
        // first node did not pass the group checks (e.g. non-contiguous dst): plain path
        ggml_d3d12_mul_mat(dev, first->src[0], src1, first);
        return 1;
    }
    ggml_d3d12_mul_mat_group(dev, src1, mats);
    return j - i;
}

static const char * ggml_d3d12_float_type_define(ggml_type t, std::vector<std::string> & defines) {
    defines.push_back(t == GGML_TYPE_F16 ? "TYPE_F16" : "TYPE_F32");
    if (t == GGML_TYPE_F16) {
        defines.push_back("USE_16BIT");
    }
    return nullptr;
}

// rms_norm, optionally fused with a following MUL by `wgt` (then `dst` is the MUL node and `norm` the RMS_NORM node)
static void ggml_d3d12_rms_norm(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * norm, ggml_tensor * dst, ggml_tensor * wgt) {
    std::vector<std::string> defines;
    if (wgt) {
        defines.push_back("FUSE_MUL");
    }
    if (norm->op == GGML_OP_L2_NORM) {
        defines.push_back("L2_NORM");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "rms_norm", hlsl_rms_norm, defines);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const d3d12_binding bw = wgt ? ggml_d3d12_bind_tensor(wgt) : bd;
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(norm, 0)),
        bw.elem_offset,
        wgt ? (uint32_t) wgt->ne[1] : 1u, wgt ? (uint32_t) wgt->ne[2] : 1u, wgt ? (uint32_t) wgt->ne[3] : 1u,
        wgt ? (uint32_t) (wgt->nb[1] / 4) : 0u, wgt ? (uint32_t) (wgt->nb[2] / 4) : 0u, wgt ? (uint32_t) (wgt->nb[3] / 4) : 0u,
    };
    std::vector<D3D12_GPU_VIRTUAL_ADDRESS> uavs = { bs.va, bd.va };
    if (wgt) {
        uavs.push_back(bw.va);
    }
    ggml_d3d12_dispatch(dev, pipeline, params, uavs, n_rows);
}

// ARGSORT and TOP_K share one kernel; TOP_K keeps the k largest (dst->ne[0]) in no particular order
static void ggml_d3d12_argsort(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (dst->op == GGML_OP_TOP_K || (ggml_sort_order) ggml_get_op_params_i32(dst, 0) == GGML_SORT_ORDER_DESC) {
        defines.push_back("SORT_DESC");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "argsort", hlsl_argsort, defines);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(src);
    std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset, (uint32_t) (src->nb[1] / 4), (uint32_t) src->ne[0], (uint32_t) dst->ne[0], 0, 0,
    };
    // every element is compared with its whole row: ne0^2 work per row, bounded per command list
    const uint64_t row_work = (uint64_t) src->ne[0] * (uint64_t) src->ne[0];
    const uint32_t chunk    = (uint32_t) std::max<uint64_t>(1, D3D12_FLASH_ATTN_WORK / row_work);
    const bool     big      = (uint64_t) n_rows * row_work > D3D12_FLASH_ATTN_WORK;
    for (uint32_t row0 = 0; row0 < n_rows; row0 += chunk) {
        params[5] = row0;
        params[6] = std::min(chunk, n_rows - row0);
        if (big && dev.dispatches_in_list > 0) {
            dev.n_flush_batch++;
            ggml_d3d12_submit_and_wait(dev);
            ggml_d3d12_begin(dev, true);
        }
        ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, params[6]);
    }
}

static void ggml_d3d12_repeat(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "repeat", hlsl_repeat, {});
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], (uint32_t) src->ne[3],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_im2col(d3d12_device_ctx & dev, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * src    = dst->src[1];
    const int32_t *     op     = (const int32_t *) dst->op_params;
    const bool          is_2D  = op[6] == 1;

    std::vector<std::string> defines;
    if (dst->type == GGML_TYPE_F16) {
        defines = { "DST_F16", "USE_16BIT" };
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "im2col", hlsl_im2col, defines);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[is_2D ? 3 : 2] / 4), (uint32_t) (src->nb[is_2D ? 2 : 1] / 4),
        is_2D ? (uint32_t) (src->nb[1] / 4) : 0u, (uint32_t) (src->nb[0] / 4),
        (uint32_t) op[0], (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5],
        (uint32_t) (is_2D ? src->ne[2] : src->ne[1]), (uint32_t) (is_2D ? src->ne[1] : 1), (uint32_t) src->ne[0],
        (uint32_t) (is_2D ? kernel->ne[1] : 1), (uint32_t) kernel->ne[0],
        (uint32_t) (is_2D ? dst->ne[2] : 1), (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_upscale(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    float sf0 = (float) dst->ne[0] / src->ne[0];
    float sf1 = (float) dst->ne[1] / src->ne[1];
    const float sf2 = (float) dst->ne[2] / src->ne[2];
    const float sf3 = (float) dst->ne[3] / src->ne[3];
    float pixel_offset = 0.5f;
    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = dst->ne[0] > 1 && src->ne[0] > 1 ? (float) (dst->ne[0] - 1) / (src->ne[0] - 1) : sf0;
        sf1 = dst->ne[1] > 1 && src->ne[1] > 1 ? (float) (dst->ne[1] - 1) / (src->ne[1] - 1) : sf1;
    }
    std::vector<std::string> defines;
    if ((mode_flags & 0xFF) == GGML_SCALE_MODE_BILINEAR) {
        defines.push_back("BILINEAR");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "upscale", hlsl_upscale, defines);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / 4), (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1],
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], ne,
        ggml_d3d12_u32_from_f32(sf0), ggml_d3d12_u32_from_f32(sf1), ggml_d3d12_u32_from_f32(sf2), ggml_d3d12_u32_from_f32(sf3),
        ggml_d3d12_u32_from_f32(pixel_offset),
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_pool_2d(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    const int32_t * op = (const int32_t *) dst->op_params;
    std::vector<std::string> defines;
    if (src->type == GGML_TYPE_F16) {
        defines.push_back("SRC_F16");
    }
    if ((ggml_op_pool) op[0] == GGML_OP_POOL_MAX) {
        defines.push_back("POOL_MAX");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "pool_2d", hlsl_pool_2d, defines);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) dst->ne[0], (uint32_t) dst->ne[1],
        (uint32_t) op[1], (uint32_t) op[2], (uint32_t) op[3], (uint32_t) op[4], (uint32_t) op[5], (uint32_t) op[6],
        ne,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

// FILL: the memset kernel with the constant's bit pattern (f16: the half value twice per word)
static void ggml_d3d12_fill(d3d12_device_ctx & dev, ggml_tensor * dst) {
    const float c = ggml_get_op_params_f32(dst, 0);
    uint32_t pattern;
    if (dst->type == GGML_TYPE_F16) {
        const uint32_t h = ggml_fp32_to_fp16(c);
        pattern = h | (h << 16);
    } else {
        pattern = ggml_d3d12_u32_from_f32(c);
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "memset", hlsl_memset,
                                                        { "BYTES_PER_THREAD=" + std::to_string(D3D12_MEMSET_BYTES_PER_THREAD) });
    const size_t   offset = ggml_d3d12_tensor_offset(dst);
    const size_t   base   = offset & ~((size_t) D3D12_BINDING_ALIGNMENT - 1);
    const uint32_t rel    = (uint32_t) (offset - base);
    const uint32_t n      = (uint32_t) ggml_nbytes(dst);
    const uint32_t threads = CEIL_DIV(rel + n, (uint32_t) D3D12_MEMSET_BYTES_PER_THREAD);
    ggml_d3d12_dispatch(dev, pipeline, { rel, n, pattern }, { ggml_d3d12_tensor_va(dst) + base },
                        CEIL_DIV(threads, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_sum_rows(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "sum_rows", hlsl_sum_rows, {});
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(src);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2], n_rows,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

static void ggml_d3d12_add_id(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * ids,
                              ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "add_id", hlsl_add_id, {});
    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(ids);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (ids->nb[1] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, bi.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_norm(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "norm", hlsl_norm, {});
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t n_rows  = (uint32_t) ggml_nrows(dst);
    const std::vector<uint32_t> params = {
        bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / 4), (uint32_t) (src->nb[2] / 4), (uint32_t) (src->nb[3] / 4),
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], n_rows,
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, n_rows);
}

// RMS_NORM at node i followed by a MUL with a broadcastable f32 weight: one dispatch; returns nodes consumed
static int ggml_d3d12_encode_rms_norm(d3d12_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * norm = cgraph->nodes[i];
    if (!dev.no_fuse && ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        ggml_tensor * w   = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        bool ok = w != norm && w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) && w->ne[0] == norm->ne[0] &&
                  mul->type == GGML_TYPE_F32 && (mul->src[0] == norm || ggml_are_same_shape(norm, w));
        for (int d = 1; d < 4; d++) {
            ok = ok && (w->ne[d] == 1 || w->ne[d] == norm->ne[d]);
        }
        if (ok) {
            ggml_d3d12_rms_norm(dev, norm->src[0], norm, mul, w);
            return 2;
        }
    }
    ggml_d3d12_rms_norm(dev, norm->src[0], norm, norm, nullptr);
    return 1;
}

static void ggml_d3d12_soft_max(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * dst) {
    std::vector<std::string> defines;
    if (src1) {
        defines.push_back("HAS_MASK");
        defines.push_back(src1->type == GGML_TYPE_F16 ? "MASK_F16" : "MASK_F32");
    }
    if (src2) {
        defines.push_back("HAS_SINK");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "soft_max", hlsl_soft_max, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = src1 ? ggml_d3d12_bind_tensor(src1) : b0;
    const d3d12_binding b2 = src2 ? ggml_d3d12_bind_tensor(src2) : b0;
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        t1 = src1 ? ggml_type_size(src1->type) : 4;

    const float max_bias    = ggml_get_op_params_f32(dst, 1);
    const float n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) src0->ne[2])));
    const float m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint32_t n_rows   = (uint32_t) ggml_nrows(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, src1 ? b1.elem_offset : 0u, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        src1 ? (uint32_t) (src1->nb[1] / t1) : 0u, src1 ? (uint32_t) (src1->nb[2] / t1) : 0u, src1 ? (uint32_t) (src1->nb[3] / t1) : 0u,
        (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        src1 ? (uint32_t) src1->ne[2] : 1u, src1 ? (uint32_t) src1->ne[3] : 1u,
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 0)),
        ggml_d3d12_u32_from_f32(max_bias), ggml_d3d12_u32_from_f32(n_head_log2),
        ggml_d3d12_u32_from_f32(m0), ggml_d3d12_u32_from_f32(m1),
        n_rows,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, b2.va, bd.va }, n_rows);
}

static void ggml_d3d12_concat(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "concat", hlsl_concat, {});

    const d3d12_binding b0  = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1  = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bd  = ggml_d3d12_bind_tensor(dst);
    const int32_t       dim = ggml_get_op_params_i32(dst, 0);
    const uint32_t      ne  = (uint32_t) ggml_nelements(dst);

    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[0] / 4), (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src0->nb[3] / 4),
        (uint32_t) (src1->nb[0] / 4), (uint32_t) (src1->nb[1] / 4), (uint32_t) (src1->nb[2] / 4), (uint32_t) (src1->nb[3] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4), (uint32_t) (dst->nb[3] / 4),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) dim, (uint32_t) src0->ne[dim],
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_ssm_conv(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "ssm_conv", hlsl_ssm_conv, {});

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);

    const std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, bd.elem_offset,
        (uint32_t) (src0->nb[1] / 4), (uint32_t) (src0->nb[2] / 4), (uint32_t) (src1->nb[1] / 4),
        (uint32_t) (dst->nb[0] / 4), (uint32_t) (dst->nb[1] / 4), (uint32_t) (dst->nb[2] / 4),
        (uint32_t) src1->ne[0], (uint32_t) src0->ne[1], (uint32_t) dst->ne[1], ne,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}


static void ggml_d3d12_flash_attn_ext(d3d12_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q     = dst->src[0];
    ggml_tensor * k     = dst->src[1];
    ggml_tensor * v     = dst->src[2];
    ggml_tensor * mask  = dst->src[3];
    ggml_tensor * sinks = dst->src[4];

    float       scale         = ggml_get_op_params_f32(dst, 0);
    const float max_bias      = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    std::vector<std::string> defines = {
        "DK=" + std::to_string(k->ne[0]), "DV=" + std::to_string(v->ne[0]),
        k->type == GGML_TYPE_F16 ? "K_F16" : k->type == GGML_TYPE_Q8_0 ? "K_Q8_0" : "K_F32",
        v->type == GGML_TYPE_F16 ? "V_F16" : v->type == GGML_TYPE_Q8_0 ? "V_Q8_0" : "V_F32",
    };
    if (mask) {
        defines.push_back("HAS_MASK");
    }
    if (sinks) {
        defines.push_back("HAS_SINKS");
    }
    if (logit_softcap != 0.0f) {
        defines.push_back("SOFTCAP");
    }
    const d3d12_binding bq = ggml_d3d12_bind_tensor(q);
    const d3d12_binding bk = ggml_d3d12_bind_tensor(k);
    const d3d12_binding bv = ggml_d3d12_bind_tensor(v);
    const d3d12_binding bm = mask ? ggml_d3d12_bind_tensor(mask) : bq;
    const d3d12_binding bs = sinks ? ggml_d3d12_bind_tensor(sinks) : bq;
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        tk = ggml_type_size(k->type);
    const size_t        tv = ggml_type_size(v->type);

    // f16 rows start on 4-byte boundaries when the element offset and every stride are even
    auto f16_aligned = [](const d3d12_binding & b, const ggml_tensor * t) {
        return b.elem_offset % 2 == 0 && (t->nb[1] / 2) % 2 == 0 && (t->nb[2] / 2) % 2 == 0 && (t->nb[3] / 2) % 2 == 0;
    };
    if (k->type == GGML_TYPE_F16 && f16_aligned(bk, k)) {
        defines.push_back("K_ALIGNED");
    }
    if (v->type == GGML_TYPE_F16 && f16_aligned(bv, v)) {
        defines.push_back("V_ALIGNED");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "flash_attn", hlsl_flash_attn, defines);

    const uint32_t n_head      = (uint32_t) q->ne[2];
    const float    n_head_log2 = (float) (1u << (uint32_t) floor(log2((double) n_head)));
    const float    m0          = powf(2.0f, -(max_bias) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, mask ? bm.elem_offset : 0u, sinks ? bs.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / tk), (uint32_t) (k->nb[2] / tk), (uint32_t) (k->nb[3] / tk),
        (uint32_t) (v->nb[1] / tv), (uint32_t) (v->nb[2] / tv), (uint32_t) (v->nb[3] / tv),
        mask ? (uint32_t) (mask->nb[1] / 2) : 0u, mask ? (uint32_t) (mask->nb[2] / 2) : 0u, mask ? (uint32_t) (mask->nb[3] / 2) : 0u,
        mask ? (uint32_t) mask->ne[2] : 1u, mask ? (uint32_t) mask->ne[3] : 1u,
        (uint32_t) q->ne[1], n_head, (uint32_t) k->ne[1],
        (uint32_t) (q->ne[2] / k->ne[2]), (uint32_t) (q->ne[3] / k->ne[3]),
        (uint32_t) (q->ne[2] / v->ne[2]), (uint32_t) (q->ne[3] / v->ne[3]),
        ggml_d3d12_u32_from_f32(scale), ggml_d3d12_u32_from_f32(max_bias), ggml_d3d12_u32_from_f32(logit_softcap),
        ggml_d3d12_u32_from_f32(n_head_log2), ggml_d3d12_u32_from_f32(m0), ggml_d3d12_u32_from_f32(m1),
        D3D12_FLASH_ATTN_BLK, 0, 0, 0,   // blk_size, n_blocks, row0, n_rows
    };
    const size_t row0_idx = params.size() - 2;

    const uint32_t n_kv     = (uint32_t) k->ne[1];
    const uint32_t n_blocks = CEIL_DIV(n_kv, (uint32_t) D3D12_FLASH_ATTN_BLK);
    params[row0_idx - 1]    = n_blocks;

    const uint64_t n_rows    = (uint64_t) ggml_nrows(dst);   // dst is [DV, n_head, n_q, n_batch]
    const uint64_t row_work  = std::max<uint64_t>(1, (uint64_t) n_kv * (uint64_t) (k->ne[0] + v->ne[0]));
    const uint64_t row_bytes = (uint64_t) n_blocks * (uint64_t) (v->ne[0] + 1) * sizeof(float);
    const uint64_t tmp_rows  = std::max<uint64_t>(1, D3D12_FLASH_ATTN_TMP_MAX / row_bytes);
    const bool     big       = n_rows * row_work > dev.fa_work;

    const size_t tmp_need = (size_t) (std::min(tmp_rows, n_rows) * row_bytes);
    if (dev.fa_tmp_size < tmp_need) {
        // the old buffer may still be referenced by recorded dispatches: run them before releasing it
        if (dev.dispatches_in_list > 0) {
            ggml_d3d12_submit_and_wait(dev);
            ggml_d3d12_begin(dev, true);
        }
        size_t size = 1ull << 20;
        while (size < tmp_need) {
            size *= 2;
        }
        dev.fa_tmp      = ggml_d3d12_create_buffer(dev, size, D3D12_HEAP_TYPE_DEFAULT, L"ggml_d3d12_fa_tmp");
        dev.fa_tmp_size = size;
    }
    const D3D12_GPU_VIRTUAL_ADDRESS tmp_va = dev.fa_tmp->GetGPUVirtualAddress();

    std::vector<std::string> combine_defines = { "DV=" + std::to_string(v->ne[0]), "COMBINE" };
    if (sinks) {
        combine_defines.push_back("HAS_SINKS");
    }
    d3d12_pipeline & combine = ggml_d3d12_get_pipeline(dev, "flash_attn", hlsl_flash_attn, combine_defines);

    for (uint64_t row0 = 0, n = 0; row0 < n_rows; row0 += n) {
        n = std::min(n_rows - row0, std::max<uint64_t>(1, std::min(dev.fa_work / row_work, tmp_rows)));
        params[row0_idx]     = (uint32_t) row0;
        params[row0_idx + 1] = (uint32_t) n;
        if (big && dev.dispatches_in_list > 0) {
            // one bounded chunk per command list; the budget follows the submit time of the previous chunk
            // (target 15..60 ms, far below the Windows GPU timeout, short enough to keep the desktop responsive)
            dev.n_flush_batch++;
            const double t0 = ggml_d3d12_time_us();
            ggml_d3d12_submit_and_wait(dev);
            const double ms = (ggml_d3d12_time_us() - t0) / 1000.0;
            if (row0 > 0) {
                if (ms < 15.0) {
                    dev.fa_work = std::min<uint64_t>(dev.fa_work * 2, 1ull << 34);
                } else if (ms > 60.0) {
                    dev.fa_work = std::max<uint64_t>(dev.fa_work / 2, 1ull << 20);
                }
            }
            ggml_d3d12_begin(dev, true);
        }
        const std::vector<D3D12_GPU_VIRTUAL_ADDRESS> uavs = { bq.va, bk.va, bv.va, bm.va, bs.va, bd.va, tmp_va };
        ggml_d3d12_dispatch(dev, pipeline, params, uavs, CEIL_DIV((uint32_t) n * n_blocks, (uint32_t) D3D12_WG_SIZE));
        ggml_d3d12_dispatch(dev, combine, params, uavs, CEIL_DIV((uint32_t) n, (uint32_t) D3D12_WG_SIZE));
    }
}

static void ggml_d3d12_gated_delta_net(d3d12_device_ctx & dev, ggml_tensor * dst) {
    ggml_tensor * q = dst->src[0];
    ggml_tensor * k = dst->src[1];
    ggml_tensor * v = dst->src[2];
    ggml_tensor * g = dst->src[3];
    ggml_tensor * b = dst->src[4];
    ggml_tensor * s = dst->src[5];

    const uint32_t S_v      = (uint32_t) v->ne[0];
    const uint32_t H        = (uint32_t) v->ne[1];
    const uint32_t n_tokens = (uint32_t) v->ne[2];
    const uint32_t n_seqs   = (uint32_t) v->ne[3];
    const uint32_t K        = (uint32_t) ggml_get_op_params_i32(dst, 0);

    std::vector<std::string> defines = { "S_V=" + std::to_string(S_v) };
    if (g->ne[0] == S_v) {
        defines.push_back("KDA");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "gated_delta_net", hlsl_gated_delta_net, defines);

    const d3d12_binding bq = ggml_d3d12_bind_tensor(q);
    const d3d12_binding bk = ggml_d3d12_bind_tensor(k);
    const d3d12_binding bv = ggml_d3d12_bind_tensor(v);
    const d3d12_binding bg = ggml_d3d12_bind_tensor(g);
    const d3d12_binding bb = ggml_d3d12_bind_tensor(b);
    const d3d12_binding bs = ggml_d3d12_bind_tensor(s);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const uint32_t n_rows  = n_seqs * H * S_v;

    std::vector<uint32_t> params = {
        bq.elem_offset, bk.elem_offset, bv.elem_offset, bg.elem_offset, bb.elem_offset, bs.elem_offset, bd.elem_offset,
        (uint32_t) (q->nb[1] / 4), (uint32_t) (q->nb[2] / 4), (uint32_t) (q->nb[3] / 4),
        (uint32_t) (k->nb[1] / 4), (uint32_t) (k->nb[2] / 4), (uint32_t) (k->nb[3] / 4),
        (uint32_t) (v->nb[1] / 4), (uint32_t) (v->nb[2] / 4), (uint32_t) (v->nb[3] / 4),
        (uint32_t) (g->nb[1] / 4), (uint32_t) (g->nb[2] / 4), (uint32_t) (g->nb[3] / 4),
        (uint32_t) (b->nb[1] / 4), (uint32_t) (b->nb[2] / 4), (uint32_t) (b->nb[3] / 4),
        H, n_tokens, (uint32_t) q->ne[1], (uint32_t) k->ne[1],
        (uint32_t) (v->ne[3] / q->ne[3]), (uint32_t) (v->ne[3] / k->ne[3]),
        (uint32_t) (s->nb[3] / 4), K,
        0, 0,   // t0, t1
        n_rows, ggml_d3d12_u32_from_f32(1.0f / sqrtf((float) S_v)),
    };
    const size_t t0_idx = 30;

    // with one snapshot the state carries over between token ranges, so long prompts are split to bound the
    // work per command list (same budget as flash attention)
    const uint64_t tok_work = std::max<uint64_t>(1, (uint64_t) n_rows * 3 * S_v);
    const uint32_t chunk    = K == 1 ? (uint32_t) std::max<uint64_t>(1, D3D12_FLASH_ATTN_WORK / tok_work) : n_tokens;
    const bool     big      = (uint64_t) n_tokens * tok_work > D3D12_FLASH_ATTN_WORK;
    for (uint32_t t0 = 0; t0 < n_tokens; t0 += chunk) {
        params[t0_idx]     = t0;
        params[t0_idx + 1] = std::min(n_tokens, t0 + chunk);
        if (big && dev.dispatches_in_list > 0) {
            dev.n_flush_batch++;
            ggml_d3d12_submit_and_wait(dev);
            ggml_d3d12_begin(dev, true);
        }
        ggml_d3d12_dispatch(dev, pipeline, params, { bq.va, bk.va, bv.va, bg.va, bb.va, bs.va, bd.va },
                            CEIL_DIV(n_rows, (uint32_t) D3D12_WG_SIZE));
    }
}

static void ggml_d3d12_rope(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * src2, ggml_tensor * dst) {
    std::vector<std::string> defines;
    ggml_d3d12_float_type_define(dst->type, defines);
    if (src2) {
        defines.push_back("FF_FUNC");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "rope", hlsl_rope, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = ggml_d3d12_bind_tensor(src1);
    const d3d12_binding b2 = src2 ? ggml_d3d12_bind_tensor(src2) : b0;
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src0->type);

    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7,  sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    int sections[4];
    memcpy(sections, (int32_t *) dst->op_params + 11, 4 * sizeof(int));
    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const uint32_t n_threads = (uint32_t) (ggml_nelements(src0) / 2);
    std::vector<uint32_t> params = {
        b0.elem_offset, b1.elem_offset, src2 ? b2.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        n_threads, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], (uint32_t) src0->ne[2],
        (uint32_t) n_dims, (uint32_t) mode,
        ggml_d3d12_u32_from_f32(theta_scale), ggml_d3d12_u32_from_f32(attn_factor),
        ggml_d3d12_u32_from_f32(freq_scale), ggml_d3d12_u32_from_f32(ext_factor),
        ggml_d3d12_u32_from_f32(corr_dims[0]), ggml_d3d12_u32_from_f32(corr_dims[1]),
        (uint32_t) sections[0], (uint32_t) sections[1], (uint32_t) sections[2], (uint32_t) sections[3],
        (uint32_t) n_offs,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, b2.va, bd.va }, CEIL_DIV(n_threads, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_glu(d3d12_device_ctx & dev, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst) {
    std::vector<std::string> defines;
    ggml_d3d12_float_type_define(dst->type, defines);
    defines.push_back(std::string("OP_") + ggml_glu_op_name(ggml_get_glu_op(dst)));
    if (!src1) {
        defines.push_back("NO_SPLIT");
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "glu", hlsl_glu, defines);

    const d3d12_binding b0 = ggml_d3d12_bind_tensor(src0);
    const d3d12_binding b1 = src1 ? ggml_d3d12_bind_tensor(src1) : b0;
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(dst->type);
    const ggml_tensor * s1 = src1 ? src1 : src0;
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);

    std::vector<uint32_t> params = {
        b0.elem_offset, src1 ? b1.elem_offset : 0u, bd.elem_offset,
        (uint32_t) (src0->nb[1] / ts), (uint32_t) (src0->nb[2] / ts), (uint32_t) (src0->nb[3] / ts),
        (uint32_t) (s1->nb[1] / ts), (uint32_t) (s1->nb[2] / ts), (uint32_t) (s1->nb[3] / ts),
        (uint32_t) (dst->nb[1] / ts), (uint32_t) (dst->nb[2] / ts), (uint32_t) (dst->nb[3] / ts),
        ne, (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],
        (uint32_t) ((int32_t *) dst->op_params)[1],
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 2)),
        ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 3)),
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { b0.va, b1.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_unary(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * dst) {
    std::vector<std::string> defines;
    ggml_d3d12_float_type_define(dst->type, defines);
    defines.push_back(dst->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(dst)) : ggml_op_name(dst->op));
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "unary", hlsl_unary, defines);

    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const uint32_t      ne = (uint32_t) ggml_nelements(dst);
    const bool          clamp = dst->op == GGML_OP_CLAMP;

    std::vector<uint32_t> params = {
        ne, bs.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[0] / ts), (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) src->ne[0], (uint32_t) src->ne[1], (uint32_t) src->ne[2],
        clamp ? ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 0)) : 0u,
        clamp ? ggml_d3d12_u32_from_f32(ggml_get_op_params_f32(dst, 1)) : 0u,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bd.va }, CEIL_DIV(ne, (uint32_t) D3D12_WG_SIZE));
}

// GET_ROWS of a quantized source: the dequant paths of the matrix-vector kernel, TPR threads per row
static void ggml_d3d12_get_rows_quant(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    const uint32_t tpr = std::min<uint32_t>(32, dev.mm_tpr_max);
    std::string    define = "SRC0_";
    define += ggml_type_name(src->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "get_rows_q", hlsl_get_rows_q,
                                                        { define, "TPR=" + std::to_string(tpr) });
    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(idx);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);
    const uint32_t      n_rows = (uint32_t) ggml_nrows(dst);

    const std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_rows,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_rows * tpr, (uint32_t) D3D12_WG_SIZE));
}

static void ggml_d3d12_get_rows(d3d12_device_ctx & dev, ggml_tensor * src, ggml_tensor * idx, ggml_tensor * dst) {
    if (ggml_is_quantized(src->type)) {
        ggml_d3d12_get_rows_quant(dev, src, idx, dst);
        return;
    }
    std::string define = "SRC_";
    define += ggml_type_name(src->type);
    for (auto & ch : define) {
        ch = (char) toupper((unsigned char) ch);
    }
    d3d12_pipeline & pipeline = ggml_d3d12_get_pipeline(dev, "get_rows", hlsl_get_rows, { define });

    const d3d12_binding bs = ggml_d3d12_bind_tensor(src);
    const d3d12_binding bi = ggml_d3d12_bind_tensor(idx);
    const d3d12_binding bd = ggml_d3d12_bind_tensor(dst);
    const size_t        ts = ggml_type_size(src->type);
    const size_t        td = ggml_type_size(dst->type);
    const bool          quant   = ggml_is_quantized(src->type);
    const uint32_t      n_units = (uint32_t) (quant ? ggml_nrows(dst) * (dst->ne[0] / 32) : ggml_nelements(dst));

    std::vector<uint32_t> params = {
        bs.elem_offset, bi.elem_offset, bd.elem_offset,
        (uint32_t) (src->nb[1] / ts), (uint32_t) (src->nb[2] / ts), (uint32_t) (src->nb[3] / ts),
        (uint32_t) (idx->nb[0] / 4), (uint32_t) (idx->nb[1] / 4), (uint32_t) (idx->nb[2] / 4),
        (uint32_t) (dst->nb[1] / td), (uint32_t) (dst->nb[2] / td), (uint32_t) (dst->nb[3] / td),
        (uint32_t) dst->ne[0], (uint32_t) idx->ne[0], (uint32_t) idx->ne[1], n_units,
    };
    ggml_d3d12_dispatch(dev, pipeline, params, { bs.va, bi.va, bd.va }, CEIL_DIV(n_units, (uint32_t) D3D12_WG_SIZE));
}

static bool ggml_d3d12_unary_supported(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_ABS: case GGML_UNARY_OP_SGN: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_ELU: case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_GELU_QUICK: case GGML_UNARY_OP_GELU_ERF: case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_HARDSWISH: case GGML_UNARY_OP_HARDSIGMOID: case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_EXPM1: case GGML_UNARY_OP_FLOOR: case GGML_UNARY_OP_CEIL: case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_TRUNC:
            return true;
        default:
            return false;
    }
}

static void ggml_d3d12_encode_node(d3d12_device_ctx & dev, ggml_tensor * node);

// encodes the node at index i, fused with following nodes when possible; returns the number of nodes consumed
static int ggml_d3d12_encode_nodes(d3d12_device_ctx & dev, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    if (!ggml_is_empty(node)) {
        if (node->op == GGML_OP_MUL_MAT) {
            return ggml_d3d12_encode_mul_mat_group(dev, cgraph, i);
        }
        if (node->op == GGML_OP_RMS_NORM) {
            return ggml_d3d12_encode_rms_norm(dev, cgraph, i);
        }
    }
    ggml_d3d12_encode_node(dev, node);
    return 1;
}

static void ggml_d3d12_encode_node(d3d12_device_ctx & dev, ggml_tensor * node) {
    if (ggml_is_empty(node)) {
        return;
    }
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            ggml_d3d12_cpy(dev, node->src[0], node);
            return;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            ggml_d3d12_binary_op(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_SCALE:
            ggml_d3d12_scale(dev, node->src[0], node);
            return;
        case GGML_OP_SET_ROWS:
            ggml_d3d12_set_rows(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT:
            ggml_d3d12_mul_mat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_MUL_MAT_ID:
            ggml_d3d12_mul_mat_id(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_RMS_NORM:
            ggml_d3d12_rms_norm(dev, node->src[0], node, node, nullptr);
            return;
        case GGML_OP_SOFT_MAX:
            ggml_d3d12_soft_max(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_d3d12_flash_attn_ext(dev, node);
            return;
        case GGML_OP_CONCAT:
            ggml_d3d12_concat(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_NORM:
            ggml_d3d12_norm(dev, node->src[0], node);
            return;
        case GGML_OP_L2_NORM:
            ggml_d3d12_rms_norm(dev, node->src[0], node, node, nullptr);
            return;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            ggml_d3d12_argsort(dev, node->src[0], node);
            return;
        case GGML_OP_REPEAT:
            ggml_d3d12_repeat(dev, node->src[0], node);
            return;
        case GGML_OP_IM2COL:
            ggml_d3d12_im2col(dev, node);
            return;
        case GGML_OP_UPSCALE:
            ggml_d3d12_upscale(dev, node->src[0], node);
            return;
        case GGML_OP_POOL_2D:
            ggml_d3d12_pool_2d(dev, node->src[0], node);
            return;
        case GGML_OP_FILL:
            ggml_d3d12_fill(dev, node);
            return;
        case GGML_OP_SUM_ROWS:
            ggml_d3d12_sum_rows(dev, node->src[0], node);
            return;
        case GGML_OP_ADD_ID:
            ggml_d3d12_add_id(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_GATED_DELTA_NET:
            ggml_d3d12_gated_delta_net(dev, node);
            return;
        case GGML_OP_SSM_CONV:
            ggml_d3d12_ssm_conv(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_ROPE:
            ggml_d3d12_rope(dev, node->src[0], node->src[1], node->src[2], node);
            return;
        case GGML_OP_GLU:
            ggml_d3d12_glu(dev, node->src[0], node->src[1], node);
            return;
        case GGML_OP_UNARY:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            ggml_d3d12_unary(dev, node->src[0], node);
            return;
        case GGML_OP_GET_ROWS:
            ggml_d3d12_get_rows(dev, node->src[0], node->src[1], node);
            return;
        default:
            GGML_ABORT("ggml_d3d12: unsupported op %s", ggml_op_name(node->op));
    }
}

/* GGML Backend Interface */

static const char * ggml_backend_d3d12_name(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_d3d12_context *) backend->context;
    return ctx->name.c_str();
}

static void ggml_d3d12_print_stats(d3d12_device_ctx & dev) {
    fprintf(stderr, "ggml_d3d12 stats [%s]: graphs %llu, nodes %llu, dispatches %llu, submits %llu (mid-graph flushes: "
                    "batch %llu, arena %llu) | graph_compute %.1f ms = encode %.1f (shader compile %.1f in %llu pipelines) + submit calls %.1f + fence wait %.1f | "
                    "set_tensor %llu calls %.1f MB %.1f ms | get_tensor %llu calls %.1f MB %.1f ms\n",
            dev.name.c_str(), (unsigned long long) dev.n_graphs, (unsigned long long) dev.n_nodes,
            (unsigned long long) dev.n_dispatches, (unsigned long long) dev.n_submits,
            (unsigned long long) dev.n_flush_batch, (unsigned long long) dev.n_flush_arena,
            dev.t_graph_us / 1000.0, (dev.t_graph_us - dev.t_submit_us - dev.t_wait_us) / 1000.0,
            dev.t_compile_us / 1000.0, (unsigned long long) dev.n_compiles,
            dev.t_submit_us / 1000.0, dev.t_wait_us / 1000.0,
            (unsigned long long) dev.n_set_tensor, dev.bytes_set / 1e6, dev.t_set_us / 1000.0,
            (unsigned long long) dev.n_get_tensor, dev.bytes_get / 1e6, dev.t_get_us / 1000.0);
    for (const auto & r : dev.rejected) {
        fprintf(stderr, "ggml_d3d12 rejected [%s]: %6llu x %s\n", dev.name.c_str(), (unsigned long long) r.second, r.first.c_str());
    }
    if (dev.profile && !dev.prof.empty()) {
        std::vector<std::pair<std::string, std::pair<double, uint64_t>>> rows(dev.prof.begin(), dev.prof.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
        double total = 0;
        for (const auto & r : rows) {
            total += r.second.first;
        }
        fprintf(stderr, "ggml_d3d12 gpu time by pipeline (total %.1f ms over %llu graphs):\n", total / 1000.0,
                (unsigned long long) dev.n_graphs);
        for (const auto & r : rows) {
            fprintf(stderr, "  %9.1f ms %5.1f%% %8llu x %7.1f us  %s\n", r.second.first / 1000.0,
                    100.0 * r.second.first / total, (unsigned long long) r.second.second,
                    r.second.first / (double) r.second.second, r.first.c_str());
        }
    }
    fflush(stderr);
}

static void ggml_backend_d3d12_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_d3d12_context *) backend->context;
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_d3d12_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto *             ctx = (ggml_backend_d3d12_context *) backend->context;
    d3d12_device_ctx & dev = *ctx->dev;
    D3D12_LOG_DEBUG("graph_compute(%d nodes)\n", cgraph->n_nodes);

    std::lock_guard<std::recursive_mutex> lock(dev.mutex);
    const double t0 = ggml_d3d12_time_us();
    ggml_d3d12_begin(dev, true);
    if (dev.n_graphs < 4 || cgraph->n_nodes != dev.last_graph_nodes) {
        // a graph of a new shape: find the pipelines it needs first and compile the missing ones in parallel
        dev.collecting = true;
        for (int i = 0; i < cgraph->n_nodes;) {
            i += ggml_d3d12_encode_nodes(dev, cgraph, i);
        }
        dev.collecting = false;
        ggml_d3d12_build_pipeline_jobs(dev);
    }
    dev.last_graph_nodes = cgraph->n_nodes;
    for (int i = 0; i < cgraph->n_nodes;) {
        i += ggml_d3d12_encode_nodes(dev, cgraph, i);
    }
    ggml_d3d12_submit_and_wait(dev);
    dev.n_graphs++;
    dev.n_nodes += cgraph->n_nodes;
    dev.t_graph_us += ggml_d3d12_time_us() - t0;
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_d3d12_i = {
    /* .get_name                = */ ggml_backend_d3d12_name,
    /* .free                    = */ ggml_backend_d3d12_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_d3d12_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_d3d12_guid(void) {
    static ggml_guid guid = { 0xd3, 0xd1, 0x2b, 0xac, 0x4e, 0x6d, 0x47, 0x1a,
                              0x9c, 0x0f, 0x8a, 0x21, 0x5b, 0x77, 0xe0, 0x39 };
    return &guid;
}

/* GGML Backend Buffer Interface */

static void ggml_backend_d3d12_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (ggml_backend_d3d12_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_d3d12_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return d3d12_ptr_base;
}

static void ggml_backend_d3d12_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_d3d12_buffer_context *) buffer->context;
    ggml_d3d12_buffer_memset(*ctx->dev, ctx->va, ggml_d3d12_tensor_offset(tensor) + offset, size, value);
}

static void ggml_backend_d3d12_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto *             ctx = (ggml_backend_d3d12_buffer_context *) buffer->context;
    d3d12_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);

    if (!dev.upload_buf) {
        dev.upload_buf = ggml_d3d12_create_buffer(dev, D3D12_STAGING_SIZE, D3D12_HEAP_TYPE_UPLOAD, L"ggml_d3d12_upload");
        GGML_ASSERT(dev.upload_buf);
        D3D12_RANGE no_read = { 0, 0 };
        ggml_d3d12_check(dev.upload_buf->Map(0, &no_read, &dev.upload_ptr), "ID3D12Resource::Map (upload)");
    }

    const double t0         = ggml_d3d12_time_us();
    const size_t dst_offset = ggml_d3d12_tensor_offset(tensor) + offset;
    size_t       done       = 0;
    while (done < size) {
        const size_t n = std::min(size - done, (size_t) D3D12_STAGING_SIZE);
        memcpy(dev.upload_ptr, (const char *) data + done, n);
        ggml_d3d12_begin(dev, false);
        dev.cmd_list->CopyBufferRegion(ctx->res.get(), dst_offset + done, dev.upload_buf.get(), 0, n);
        ggml_d3d12_submit_and_wait(dev);
        done += n;
    }
    dev.n_set_tensor++;
    dev.bytes_set += size;
    dev.t_set_us += ggml_d3d12_time_us() - t0;
}

static void ggml_backend_d3d12_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto *             ctx = (ggml_backend_d3d12_buffer_context *) buffer->context;
    d3d12_device_ctx & dev = *ctx->dev;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(dev.mutex);

    if (!dev.readback_buf) {
        dev.readback_buf = ggml_d3d12_create_buffer(dev, D3D12_STAGING_SIZE, D3D12_HEAP_TYPE_READBACK, L"ggml_d3d12_readback");
        GGML_ASSERT(dev.readback_buf);
    }

    const double t0         = ggml_d3d12_time_us();
    const size_t src_offset = ggml_d3d12_tensor_offset(tensor) + offset;
    size_t       done       = 0;
    while (done < size) {
        const size_t n = std::min(size - done, (size_t) D3D12_STAGING_SIZE);
        ggml_d3d12_begin(dev, false);
        dev.cmd_list->CopyBufferRegion(dev.readback_buf.get(), 0, ctx->res.get(), src_offset + done, n);
        ggml_d3d12_submit_and_wait(dev);

        void *      ptr   = nullptr;
        D3D12_RANGE range = { 0, n };
        ggml_d3d12_check(dev.readback_buf->Map(0, &range, &ptr), "ID3D12Resource::Map (readback)");
        memcpy((char *) data + done, ptr, n);
        D3D12_RANGE no_write = { 0, 0 };
        dev.readback_buf->Unmap(0, &no_write);
        done += n;
    }
    dev.n_get_tensor++;
    dev.bytes_get += size;
    dev.t_get_us += ggml_d3d12_time_us() - t0;
}

static void ggml_backend_d3d12_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_d3d12_buffer_context *) buffer->context;
    ggml_d3d12_buffer_memset(*ctx->dev, ctx->va, 0, ctx->size, value);
}

static ggml_backend_buffer_i ggml_backend_d3d12_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_d3d12_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_d3d12_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_d3d12_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_d3d12_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_d3d12_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_d3d12_buffer_clear,
    /* .reset           = */ NULL,
};

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_d3d12_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * dev = (d3d12_device_ctx *) buft->context;
    return dev->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_d3d12_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev = (d3d12_device_ctx *) buft->context;
    std::lock_guard<std::recursive_mutex> lock(dev->mutex);

    const size_t alloc_size = std::max((size_t) D3D12_BINDING_ALIGNMENT,
                                       (size + D3D12_BINDING_ALIGNMENT - 1) & ~((size_t) D3D12_BINDING_ALIGNMENT - 1))
                              // One binding alignment of slack past the end. Quant blocks whose size is
                              // not a multiple of 4 (iq4_nl and q4_0 are 18 bytes, q5_0 22, q5_1 24) end
                              // mid-word, and ByteAddressBuffer.Load only reads 4-byte aligned words, so
                              // reading the last block's tail necessarily touches a few bytes past it.
                              // Root UAVs carry no size, so that read is unbounded rather than clamped:
                              // the Radeon tolerates it, the MTT S80 faults and the device is removed.
                              // Verified by removing this slack, which brings the fault straight back.
                              + D3D12_BINDING_ALIGNMENT;
    D3D12_LOG_DEBUG("alloc_buffer(%zu bytes)\n", alloc_size);

    com_ptr<ID3D12Resource> res = ggml_d3d12_create_buffer(*dev, alloc_size, D3D12_HEAP_TYPE_DEFAULT, L"ggml_d3d12_tensor_buf");
    if (!res) {
        return nullptr;
    }
    auto * ctx = new ggml_backend_d3d12_buffer_context();
    ctx->res  = res;
    ctx->va   = res->GetGPUVirtualAddress();
    ctx->size = alloc_size;
    ctx->dev = ggml_d3d12_shared_dev(dev);
    return ggml_backend_buffer_init(buft, ggml_backend_d3d12_buffer_interface, ctx, size);
}

static size_t ggml_backend_d3d12_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return D3D12_BINDING_ALIGNMENT;
}

static size_t ggml_backend_d3d12_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * dev = (d3d12_device_ctx *) buft->context;
    return dev->max_alloc;
}

/* GGML Backend Device Interface */

static const char * ggml_backend_d3d12_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = (d3d12_device_ctx *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_d3d12_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = (d3d12_device_ctx *) dev->context;
    return ctx->desc.c_str();
}

static void ggml_backend_d3d12_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = (d3d12_device_ctx *) dev->context;
    *total     = ctx->caps.uma ? ctx->shared_mem : ctx->dedicated_mem;
    *free      = *total;
    com_ptr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(ctx->adapter->QueryInterface(IID_PPV_ARGS(adapter3.put())))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        const DXGI_MEMORY_SEGMENT_GROUP group = ctx->caps.uma ? DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL : DXGI_MEMORY_SEGMENT_GROUP_LOCAL;
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, group, &info))) {
            *free = info.Budget > info.CurrentUsage ? (size_t) (info.Budget - info.CurrentUsage) : 0;
            if (info.Budget > 0) {
                *total = (size_t) info.Budget;
            }
        }
    }
}

static enum ggml_backend_dev_type ggml_backend_d3d12_device_get_type(ggml_backend_dev_t dev) {
    auto * ctx = (d3d12_device_ctx *) dev->context;
    return ctx->caps.uma ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_d3d12_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_d3d12_device_get_name(dev);
    props->description = ggml_backend_d3d12_device_get_description(dev);
    props->type        = ggml_backend_d3d12_device_get_type(dev);
    ggml_backend_d3d12_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_d3d12_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dev_ctx = (d3d12_device_ctx *) dev->context;

    auto * ctx = new ggml_backend_d3d12_context();
    ctx->dev   = ggml_d3d12_shared_dev(dev_ctx);
    ctx->name  = dev_ctx->name;

    auto * backend = new ggml_backend();
    *backend       = {
        /* .guid      = */ ggml_backend_d3d12_guid(),
        /* .interface = */ ggml_backend_d3d12_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_d3d12_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = (d3d12_device_ctx *) dev->context;
    return &ctx->buft;
}

static bool ggml_backend_d3d12_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_d3d12_buffer_type_get_name && buft->device == dev;
}

static bool ggml_d3d12_supports_op(d3d12_device_ctx * ctx, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (!ctx->disable_ops.empty() && op->op != GGML_OP_NONE && op->op != GGML_OP_VIEW &&
        op->op != GGML_OP_RESHAPE && op->op != GGML_OP_PERMUTE && op->op != GGML_OP_TRANSPOSE) {
        const std::string name = std::string(",") + ggml_op_name(op->op) + ",";
        // ALL keeps SET_ROWS, otherwise the KV cache cannot be allocated on this device at all
        const bool all = ctx->disable_ops == ",ALL," && op->op != GGML_OP_SET_ROWS;
        if (all || ctx->disable_ops.find(name) != std::string::npos) {
            return false;
        }
    }

    auto type_ok = [&](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_I32 || (t == GGML_TYPE_F16 && ctx->caps.native_16bit);
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return type_ok(op->type) && type_ok(src0->type);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == op->type && src1->type == op->type;
        case GGML_OP_SCALE:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32;
        case GGML_OP_SET_ROWS:
            if (op->type == GGML_TYPE_Q8_0) {
                // rows are quantized in place: contiguous dst, contiguous source rows
                return src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32) &&
                       ggml_is_contiguous(op) && src0->nb[0] == sizeof(float) && op->ne[0] % 32 == 0 &&
                       ggml_nbytes(op) < (1ull << 31);
            }
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == GGML_TYPE_F32 && (src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);
        case GGML_OP_MUL_MAT:
            {
                // mat-vec kernel, any column count in chunks of 4; contiguous rows required
                const bool quant = ggml_is_quantized(src0->type);
                return ggml_d3d12_mul_mat_vec_type(src0->type) &&
                       (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) && op->type == GGML_TYPE_F32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
                       (quant ? src0->ne[0] % 256 == 0 || (src0->ne[0] % 32 == 0 && ggml_blck_size(src0->type) == 32)
                              : src0->ne[0] % 4 == 0);
            }
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // the rank kernel is quadratic in the row length: long rows (e.g. vocab sorts) stay on the CPU
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                   ggml_is_contiguous_rows(src0) && src0->ne[0] <= 1024;
        case GGML_OP_REPEAT:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_IM2COL:
            return src1->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                   ggml_is_contiguous(op) && ggml_nelements(op) <= UINT32_MAX && ggml_nelements(src1) <= UINT32_MAX;
        case GGML_OP_POOL_2D: {
            const ggml_op_pool pool = (ggml_op_pool) ggml_get_op_params_i32(op, 0);
            return op->type == GGML_TYPE_F32 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
                   (pool == GGML_OP_POOL_AVG || pool == GGML_OP_POOL_MAX) && ggml_is_contiguous(op) &&
                   src0->nb[0] == ggml_type_size(src0->type) && ggml_nelements(op) <= UINT32_MAX;
        }
        case GGML_OP_UPSCALE: {
            const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
            const int32_t mode       = mode_flags & 0xFF;
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && !(mode_flags & GGML_SCALE_FLAG_ANTIALIAS) &&
                   (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR) &&
                   ggml_nelements(op) <= UINT32_MAX && ggml_nelements(src0) <= UINT32_MAX;
        }
        case GGML_OP_FILL:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous(op) &&
                   ggml_nbytes(op) < (1ull << 30);
        case GGML_OP_SUM_ROWS:
            return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
        case GGML_OP_ADD_ID:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_I32 && ggml_is_contiguous(src1) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_MUL_MAT_ID:
            {
                const ggml_tensor * ids = op->src[2];
                const bool quant = ggml_is_quantized(src0->type);
                return ggml_d3d12_mul_mat_vec_type(src0->type) && src1->type == GGML_TYPE_F32 &&
                       op->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
                       src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) &&
                       (quant ? src0->ne[0] % 256 == 0 || (src0->ne[0] % 32 == 0 && ggml_blck_size(src0->type) == 32)
                              : src0->ne[0] % 4 == 0);
            }
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src0);
        case GGML_OP_SOFT_MAX:
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 &&
                   (!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_GATED_DELTA_NET:
            {
                // f32 everywhere, rows of S_v (a multiple of 4, <= 512) elements, q/k head size equal to S_v
                const ggml_tensor * v = op->src[2];
                bool ok = op->type == GGML_TYPE_F32 && v->ne[0] % 4 == 0 && v->ne[0] <= 512 &&
                          op->src[0]->ne[0] == v->ne[0] && op->src[1]->ne[0] == v->ne[0] &&
                          ggml_nelements(op) <= UINT32_MAX;
                for (int i = 0; i < 6; i++) {
                    ok = ok && op->src[i]->type == GGML_TYPE_F32 && op->src[i]->nb[0] == sizeof(float);
                }
                return ok;
            }
        case GGML_OP_SSM_CONV:
            // same layout requirements as the CPU kernel
            return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                   src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) &&
                   src0->nb[1] == src0->ne[0] * sizeof(float) && src1->nb[1] == src1->ne[0] * sizeof(float) &&
                   ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_CONCAT:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) && src0->type == op->type &&
                   src1->type == op->type && ggml_nelements(op) <= UINT32_MAX;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // f32/f16 KV, f16 mask, head sizes up to 576 and multiples of 4; contiguous rows and a contiguous dst
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];
                const ggml_tensor * m = op->src[3];
                const ggml_tensor * s = op->src[4];
                auto kv_ok = [](const ggml_tensor * t) {
                    if (t->type == GGML_TYPE_Q8_0) {
                        return t->ne[0] <= 576 && t->ne[0] % 32 == 0 && t->nb[1] % ggml_type_size(t->type) == 0 &&
                               t->nb[2] % ggml_type_size(t->type) == 0 && t->nb[3] % ggml_type_size(t->type) == 0;
                    }
                    return (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && t->nb[0] == ggml_type_size(t->type) &&
                           t->ne[0] <= 576 && t->ne[0] % 4 == 0;
                };
                return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && src0->nb[0] == sizeof(float) &&
                       kv_ok(k) && kv_ok(v) && (!m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2)) &&
                       (!s || s->type == GGML_TYPE_F32) && ggml_is_contiguous(op);
            }
        case GGML_OP_ROPE:
            return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                   src0->type == op->type && src1->type == GGML_TYPE_I32 && src0->ne[0] % 2 == 0 &&
                   (!op->src[2] || op->src[2]->type == GGML_TYPE_F32);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK: case GGML_GLU_OP_SWIGLU_CLAMP:
                    return (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit)) &&
                           src0->type == op->type && (!src1 || src1->type == op->type);
                case GGML_GLU_OP_SWIGLU_OAI:
                    return op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32 && (!src1 || src1->type == GGML_TYPE_F32);
                default:
                    return false;
            }
        case GGML_OP_UNARY:
            return ggml_d3d12_unary_supported(ggml_get_unary_op(op)) && src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit));
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
            return src0->type == op->type &&
                   (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_F16 && ctx->caps.native_16bit));
        case GGML_OP_GET_ROWS:
            if (src1->type != GGML_TYPE_I32) {
                return false;
            }
            switch (src0->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return op->type == GGML_TYPE_F32;
                case GGML_TYPE_I32:
                    return op->type == GGML_TYPE_I32;
                default:
                    // quantized sources go through the dequant paths of the matrix-vector kernel
                    return ggml_is_quantized(src0->type) && ggml_d3d12_mul_mat_vec_type(src0->type) &&
                           op->type == GGML_TYPE_F32 && op->ne[0] % ggml_blck_size(src0->type) == 0;
            }
        default:
            return false;
    }
}

static bool ggml_backend_d3d12_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto *     ctx = (d3d12_device_ctx *) dev->context;
    const bool ok  = ggml_d3d12_supports_op(ctx, op);
    if (!ok && ctx->stats) {
        std::string key = ggml_op_desc(op);
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            key += (i == 0 ? " " : ",");
            key += ggml_type_name(op->src[i]->type);
        }
        key += std::string(" -> ") + ggml_type_name(op->type);
        std::lock_guard<std::mutex> lock(ctx->rejected_mutex);
        ctx->rejected[key]++;
    }
    return ok;
}

static struct ggml_backend_device_i ggml_backend_d3d12_device_i = {
    /* .get_name             = */ ggml_backend_d3d12_device_get_name,
    /* .get_description      = */ ggml_backend_d3d12_device_get_description,
    /* .get_memory           = */ ggml_backend_d3d12_device_get_memory,
    /* .get_type             = */ ggml_backend_d3d12_device_get_type,
    /* .get_props            = */ ggml_backend_d3d12_device_get_props,
    /* .init_backend         = */ ggml_backend_d3d12_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_d3d12_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_d3d12_device_supports_op,
    /* .supports_buft        = */ ggml_backend_d3d12_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* Registry: adapter enumeration and device initialization */

struct ggml_backend_d3d12_reg_context {
    std::vector<std::shared_ptr<d3d12_device_ctx>> devs;
    std::vector<ggml_backend_device>               devices;
};

static ggml_backend_d3d12_reg_context * g_reg_ctx = nullptr;

static void ggml_d3d12_atexit() {
    if (!g_reg_ctx) {
        return;
    }
    for (auto & dev : g_reg_ctx->devs) {
        if (dev->stats) {
            ggml_d3d12_print_stats(*dev);
        }
    }
}

static std::shared_ptr<d3d12_device_ctx> ggml_d3d12_shared_dev(d3d12_device_ctx * dev) {
    for (auto & d : g_reg_ctx->devs) {
        if (d.get() == dev) {
            return d;
        }
    }
    GGML_ABORT("ggml_d3d12: unknown device context");
}

static bool ggml_d3d12_init_device(d3d12_device_ctx & dev, ggml_backend_dev_t ggml_dev) {
    HRESULT hr;

    // capabilities
    D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_7 };
    while (FAILED(dev.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
           sm.HighestShaderModel > D3D_SHADER_MODEL_6_0) {
        sm.HighestShaderModel = (D3D_SHADER_MODEL) (sm.HighestShaderModel - 1);
    }
    dev.caps.shader_model = sm.HighestShaderModel;
    if (const char * env = getenv("GGML_D3D12_SM")) {
        // e.g. GGML_D3D12_SM=60 caps the shader model used for kernels
        int v = atoi(env);
        if (v >= 60 && v <= 67) {
            D3D_SHADER_MODEL capped = (D3D_SHADER_MODEL) (0x60 + (v - 60));
            dev.caps.shader_model   = std::min(dev.caps.shader_model, capped);
        }
    }
    if (dev.caps.shader_model < D3D_SHADER_MODEL_6_0) {
        GGML_LOG_WARN("ggml_d3d12: %s does not support shader model 6.0, skipping\n", dev.desc.c_str());
        return false;
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
    if (SUCCEEDED(dev.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1)))) {
        dev.caps.wave_ops = o1.WaveOps;
        dev.caps.wave_min = o1.WaveLaneCountMin;
        dev.caps.wave_max = o1.WaveLaneCountMax;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4 = {};
    if (SUCCEEDED(dev.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4)))) {
        dev.caps.native_16bit = o4.Native16BitShaderOpsSupported && dev.caps.shader_model >= D3D_SHADER_MODEL_6_2;
    }
    D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
    if (SUCCEEDED(dev.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)))) {
        dev.caps.uma = arch.UMA;
    }
    if (const char * env = getenv("GGML_D3D12_MAX_ALLOC_MB")) {
        dev.max_alloc = (size_t) atoll(env) * 1024 * 1024;
    }
    if (const char * env = getenv("GGML_D3D12_SUBMIT_BATCH")) {
        dev.submit_batch = std::max(1, atoi(env));
    }
    if (getenv("GGML_D3D12_NO_FUSE") != nullptr) {
        dev.no_fuse = true;
    }
    if (const char * env = getenv("GGML_D3D12_MM_TPR")) {
        dev.mm_tpr_max = (uint32_t) std::max(1, atoi(env));
    }
    if (const char * env = getenv("GGML_D3D12_TILED")) {
        // column count from which the tiled prompt kernel takes over; 1 means "always when eligible"
        dev.tiled_min_cols = (uint32_t) std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_D3D12_DISABLE_OPS")) {
        dev.disable_ops = std::string(",") + env + ",";
        // the runner cannot pass commas, so dots separate names too; ALL refuses every op (CPU only)
        for (char & c : dev.disable_ops) {
            if (c == '.') { c = ','; }
        }
    }
#ifdef GGML_D3D12_FORCE_NO_BARRIER
    if (true) {
#else
    if (getenv("GGML_D3D12_NO_BARRIER") != nullptr) {
#endif
        // timing experiments only: without UAV barriers dependent dispatches overlap and results are garbage
        dev.no_barrier = true;
        GGML_LOG_WARN("ggml_d3d12: GGML_D3D12_NO_BARRIER set, results will be wrong\n");
    }
    // queue, allocator, command list, fence
    D3D12_COMMAND_QUEUE_DESC qdesc = {};
    qdesc.Type                     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qdesc.Priority                 = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    qdesc.Flags                    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = dev.device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(dev.queue.put()));
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: CreateCommandQueue failed 0x%08lx\n", (unsigned long) hr); return false; }
    hr = dev.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(dev.allocator.put()));
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: CreateCommandAllocator failed 0x%08lx\n", (unsigned long) hr); return false; }
    hr = dev.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dev.allocator.get(), nullptr, IID_PPV_ARGS(dev.cmd_list.put()));
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: CreateCommandList failed 0x%08lx\n", (unsigned long) hr); return false; }
    dev.cmd_list->Close();
    hr = dev.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(dev.fence.put()));
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: CreateFence failed 0x%08lx\n", (unsigned long) hr); return false; }
    dev.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    GGML_ASSERT(dev.fence_event);

    // root signature: b0 = params CBV, u0..u11 = raw buffer UAVs
    D3D12_ROOT_PARAMETER root_params[1 + D3D12_MAX_ROOT_UAVS] = {};
    root_params[0].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
    root_params[0].Descriptor.ShaderRegister = 0;
    root_params[0].Descriptor.RegisterSpace  = 0;
    root_params[0].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 0; i < D3D12_MAX_ROOT_UAVS; i++) {
        root_params[1 + i].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_UAV;
        root_params[1 + i].Descriptor.ShaderRegister = i;
        root_params[1 + i].Descriptor.RegisterSpace  = 0;
        root_params[1 + i].ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
    }
    D3D12_ROOT_SIGNATURE_DESC rs_desc = {};
    rs_desc.NumParameters             = 1 + D3D12_MAX_ROOT_UAVS;
    rs_desc.pParameters               = root_params;
    rs_desc.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    com_ptr<ID3DBlob> rs_blob, rs_err;
    hr = D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, rs_blob.put(), rs_err.put());
    if (FAILED(hr)) {
        GGML_LOG_ERROR("ggml_d3d12: D3D12SerializeRootSignature failed: %s\n",
                       rs_err ? (const char *) rs_err->GetBufferPointer() : "");
        return false;
    }
    hr = dev.device->CreateRootSignature(0, rs_blob->GetBufferPointer(), rs_blob->GetBufferSize(), IID_PPV_ARGS(dev.root_sig.put()));
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: CreateRootSignature failed 0x%08lx\n", (unsigned long) hr); return false; }

    // parameter arena
    dev.param_buf = ggml_d3d12_create_buffer(dev, (size_t) D3D12_PARAM_SLOT_SIZE * D3D12_PARAM_SLOT_COUNT,
                                             D3D12_HEAP_TYPE_UPLOAD, L"ggml_d3d12_params");
    if (!dev.param_buf) { return false; }
    D3D12_RANGE no_read = { 0, 0 };
    void * ptr = nullptr;
    hr = dev.param_buf->Map(0, &no_read, &ptr);
    if (FAILED(hr)) { GGML_LOG_ERROR("ggml_d3d12: Map (params) failed 0x%08lx\n", (unsigned long) hr); return false; }
    dev.param_ptr = (uint8_t *) ptr;
    dev.param_va  = dev.param_buf->GetGPUVirtualAddress();

    // shader compiler (dxcompiler.dll + dxil.dll must be next to the executable or on PATH)
    dev.dxc_module = LoadLibraryA("dxcompiler.dll");
    if (!dev.dxc_module) {
        GGML_LOG_ERROR("ggml_d3d12: dxcompiler.dll not found (place dxcompiler.dll and dxil.dll next to the executable)\n");
        return false;
    }
    auto create = (DxcCreateInstanceProc) (void *) GetProcAddress(dev.dxc_module, "DxcCreateInstance");
    if (!create || FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(dev.compiler.put()))) ||
        FAILED(create(CLSID_DxcUtils, IID_PPV_ARGS(dev.utils.put())))) {
        GGML_LOG_ERROR("ggml_d3d12: failed to create DXC compiler instance\n");
        return false;
    }
    if (!GetModuleHandleA("dxil.dll") && !LoadLibraryA("dxil.dll")) {
        GGML_LOG_WARN("ggml_d3d12: dxil.dll not found; shaders will be unsigned and most drivers reject them\n");
    }

    // stats and GPU timestamp profiling (needs the queue)
    dev.stats   = getenv("GGML_D3D12_STATS") != nullptr;
    dev.profile = getenv("GGML_D3D12_PROFILE") != nullptr;
    if (dev.profile) {
        dev.stats = true;
        D3D12_QUERY_HEAP_DESC qh = {};
        qh.Type                  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qh.Count                 = D3D12_QUERY_CAPACITY;
        if (FAILED(dev.device->CreateQueryHeap(&qh, IID_PPV_ARGS(dev.query_heap.put()))) ||
            FAILED(dev.queue->GetTimestampFrequency(&dev.timestamp_freq))) {
            GGML_LOG_WARN("ggml_d3d12: timestamp queries unavailable, profiling disabled\n");
            dev.profile = false;
        } else {
            dev.query_readback = ggml_d3d12_create_buffer(dev, (size_t) D3D12_QUERY_CAPACITY * sizeof(uint64_t),
                                                          D3D12_HEAP_TYPE_READBACK, L"ggml_d3d12_queries");
            dev.profile = (bool) dev.query_readback;
        }
    }
    if (dev.stats) {
        static bool registered = false;
        if (!registered) {
            registered = true;
            atexit(ggml_d3d12_atexit);
        }
    }

    // buffer type
    dev.buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_d3d12_buffer_type_get_name,
            /* .alloc_buffer   = */ ggml_backend_d3d12_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_d3d12_buffer_type_get_alignment,
            /* .get_max_size   = */ ggml_backend_d3d12_buffer_type_get_max_size,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ NULL,
        },
        /* .device  = */ ggml_dev,
        /* .context = */ &dev,
    };

    GGML_LOG_INFO("ggml_d3d12: %s = %s | SM %d.%d | 16-bit %s | wave %u-%u | %s | %zu MiB | build %s %s\n", dev.name.c_str(),
                  dev.desc.c_str(), dev.caps.shader_model >> 4, dev.caps.shader_model & 0xf,
                  dev.caps.native_16bit ? "yes" : "no", dev.caps.wave_min, dev.caps.wave_max,
                  dev.caps.uma ? "UMA" : "discrete", (dev.caps.uma ? dev.shared_mem : dev.dedicated_mem) / (1024 * 1024),
                  __DATE__, __TIME__);
    return true;
}

static void ggml_d3d12_enumerate(ggml_backend_d3d12_reg_context & reg_ctx, ggml_backend_reg_t reg) {
    UINT factory_flags = 0;
    if (getenv("GGML_D3D12_DEBUG") != nullptr) {
        com_ptr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.put())))) {
            debug->EnableDebugLayer();
            factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
            GGML_LOG_INFO("ggml_d3d12: debug layer enabled\n");
        } else {
            GGML_LOG_WARN("ggml_d3d12: debug layer requested but unavailable (install the Graphics Tools feature)\n");
        }
    }
    const bool allow_warp = getenv("GGML_D3D12_WARP") != nullptr;

    com_ptr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(factory.put())))) {
        GGML_LOG_WARN("ggml_d3d12: CreateDXGIFactory2 failed\n");
        return;
    }
    com_ptr<IDXGIFactory6> factory6;
    factory->QueryInterface(IID_PPV_ARGS(factory6.put()));

    std::vector<com_ptr<IDXGIAdapter1>> adapters;
    if (getenv("GGML_D3D12_DISABLE") != nullptr) {
        GGML_LOG_WARN("ggml_d3d12: GGML_D3D12_DISABLE set, exposing no devices\n");
        return;
    }
    for (UINT i = 0;; i++) {
        com_ptr<IDXGIAdapter1> adapter;
        HRESULT hr;
        if (factory6) {
            hr = factory6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(adapter.put()));
        } else {
            hr = factory->EnumAdapters1(i, adapter.put());
        }
        if (hr == DXGI_ERROR_NOT_FOUND || FAILED(hr)) {
            break;
        }
        adapters.push_back(adapter);
    }

    // first pass: create the D3D12 devices, so device contexts are stable before ggml devices point at them
    for (auto & adapter : adapters) {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && !allow_warp) {
            continue;
        }
        auto dev = std::make_shared<d3d12_device_ctx>();
        if (FAILED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(dev->device.put())))) {
            continue;
        }
        dev->adapter       = adapter;
        dev->desc          = ggml_d3d12_wide_to_utf8(desc.Description);
        dev->dedicated_mem = desc.DedicatedVideoMemory;
        dev->shared_mem    = desc.SharedSystemMemory;
        dev->name          = GGML_D3D12_NAME + std::to_string(reg_ctx.devs.size());
        reg_ctx.devs.push_back(dev);
    }

    reg_ctx.devices.reserve(reg_ctx.devs.size());
    std::vector<std::shared_ptr<d3d12_device_ctx>> kept;
    for (auto & dev : reg_ctx.devs) {
        ggml_backend_device ggml_dev = {
            /* .iface   = */ ggml_backend_d3d12_device_i,
            /* .reg     = */ reg,
            /* .context = */ dev.get(),
        };
        reg_ctx.devices.push_back(ggml_dev);
        if (!ggml_d3d12_init_device(*dev, &reg_ctx.devices.back())) {
            reg_ctx.devices.pop_back();
            continue;
        }
        kept.push_back(dev);
    }
    reg_ctx.devs = std::move(kept);
    // renumber after drops
    for (size_t i = 0; i < reg_ctx.devs.size(); i++) {
        reg_ctx.devs[i]->name = GGML_D3D12_NAME + std::to_string(i);
    }
}

static const char * ggml_backend_d3d12_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_D3D12_NAME;
}

static size_t ggml_backend_d3d12_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = (ggml_backend_d3d12_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_d3d12_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = (ggml_backend_d3d12_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_d3d12_reg_i = {
    /* .get_name         = */ ggml_backend_d3d12_reg_get_name,
    /* .get_device_count = */ ggml_backend_d3d12_reg_get_device_count,
    /* .get_device       = */ ggml_backend_d3d12_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_d3d12_reg() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // leaked on purpose: D3D12 objects must not be torn down during static destruction
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_d3d12_reg_i,
        /* .context     = */ nullptr,
    };
    if (g_reg_ctx == nullptr) {
        g_reg_ctx   = new ggml_backend_d3d12_reg_context();
        reg.context = g_reg_ctx;
        ggml_d3d12_enumerate(*g_reg_ctx, &reg);
    }
    return &reg;
}

ggml_backend_t ggml_backend_d3d12_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_d3d12_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_d3d12_device_init_backend(ggml_backend_reg_dev_get(reg, device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_d3d12_reg)
