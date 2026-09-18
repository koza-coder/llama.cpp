# Running llama.cpp on any GPU with Direct3D 12

This fork adds a native **Direct3D 12** backend to llama.cpp. It needs no CUDA, no ROCm, no
vendor SDK and no Vulkan driver - only a GPU whose Windows driver supports D3D12 feature level
11_0, which covers nearly every graphics card sold since about 2015, including ones no other
llama.cpp backend supports.

You do not need to build anything. Download the zip, download a model, run one command.

Everything below is Windows x64. Copy and paste each block into a Command Prompt or PowerShell
window. `curl` and `tar` are part of Windows 10 and 11, so nothing else has to be installed.

---

## 1. Get the release

Download the zip from the releases page and unpack it:

```
curl -L -o llama-d3d12.zip https://github.com/koza-coder/llama.cpp/releases/download/d3d12-v0.1.1/llama-d3d12-d3d12-v0.1.1-win-x64.zip
```

```
tar -xf llama-d3d12.zip
```

That creates a folder `llama-d3d12-d3d12-v0.1.1-win-x64`. Go into it:

```
cd llama-d3d12-d3d12-v0.1.1-win-x64
```

Check you got the file intact before running it - the number must match exactly:

```
certutil -hashfile ..\llama-d3d12.zip SHA256
```

```
da30a0c1c5ba16a9fbbc148bbc2e93ba9ac939875d4dc51ce6f96f6b47df86a0
```

Inside you will find four programs and two DLLs:

| file | what it is |
|---|---|
| `llama-cli.exe` | chat with a model |
| `llama-bench.exe` | measure speed |
| `test-backend-ops.exe` | check the GPU produces correct results |
| `llama-mtmd-cli.exe` | describe a picture or transcribe speech |
| `dxcompiler.dll`, `dxil.dll` | **required** - the backend compiles its shaders with these at startup |

Keep all six files together. The programs are otherwise self-contained: no runtime to install,
no other DLLs, nothing added to your system.

---

## 2. Check your GPU is found

```
llama-bench.exe --list-devices
```

You should see a line naming your graphics card with `D3D120` in front of it. If you have two
GPUs you will see `D3D120` and `D3D121`.

If nothing is listed, your driver does not expose a D3D12 device. Updating the graphics driver is
the first thing to try.

---

## 3. Get a model

Pick one. The first is small enough to try in a minute; the others are better at actually holding
a conversation. Each is one file, and the `curl` line puts it in the folder you are already in.

**Tiny, 105 MB** - SmolLM2 135M. Fast to download, talks nonsense but proves the whole path works:

```
curl -L -o model.gguf https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

**Small, 429 MB** - Qwen2.5 0.5B. A reasonable first real model:

```
curl -L -o model.gguf https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
```

Check the download is complete (`certutil -hashfile model.gguf SHA256`):

| file | bytes | SHA-256 |
|---|---:|---|
| `SmolLM2-135M-Instruct-Q4_K_M.gguf` | 105454432 | `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d` |
| `qwen2.5-0.5b-instruct-q4_0.gguf` | 428730208 | `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` |

Any `.gguf` file from Hugging Face works, not only these two. Larger models need more video
memory; a rough rule is that the file has to fit in your card's VRAM, plus about 1 GB for the
context.

---

## 4. Talk to it

```
llama-cli.exe -m model.gguf -ngl 99
```

`-ngl 99` means "put every layer on the GPU". Without it the model runs on your processor and the
GPU is never used, which is a common reason for a disappointing first run.

Type a message, press Enter. Ctrl+C quits.

For a single answer instead of a chat:

```
llama-cli.exe -m model.gguf -ngl 99 -no-cnv -p "The capital of France is"
```

---

## 5. Measure it

```
llama-bench.exe -m model.gguf -ngl 99
```

Two numbers come out. `pp512` is prompt processing, how fast it reads your input. `tg128` is text
generation, how fast it writes the answer. Both are tokens per second, higher is better.

To see what difference the GPU makes, run it again with `-ngl 0` and compare.

---

## 6. Check the GPU is correct, not just fast

This is the interesting one. It runs every operation on your GPU and compares the result against
the same operation on your processor, so it will tell you whether this backend is actually right
on hardware nobody has tested it on.

Start small:

```
test-backend-ops.exe test -o GET_ROWS
```

Then the big one, which takes a few minutes:

```
test-backend-ops.exe test -o MUL_MAT
```

Every line should end in `OK`. A line ending in `FAIL` is a real result worth reporting - see
section 8.

> **A warning worth reading.** A GPU compute program that misbehaves can freeze your screen for a
> few seconds while Windows resets the display driver, and on some drivers it can take the desktop
> down with it. Save your work before running the tests for the first time. If the screen goes
> black and comes back, that is Windows recovering and nothing is damaged. If you want to be
> careful, run the op groups one at a time rather than all at once, and do not run `perf` mode
> until the normal tests pass.

To test one device when you have more than one card, use its exact name:

```
test-backend-ops.exe test -b D3D120 -o MUL_MAT
```

---

## 7. Pictures and sound

`llama-mtmd-cli.exe` runs multimodal models. These need **two** files: the model itself and a
matching "mmproj" projector that turns a picture or a sound into something the model understands.
Both come from the same Hugging Face repository, and they must be a matching pair.

### Describe a picture

```
curl -L -o vision.gguf https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf
```

```
curl -L -o vision-mmproj.gguf https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf
```

```
llama-mtmd-cli.exe -m vision.gguf --mmproj vision-mmproj.gguf --image picture.jpg -ngl 99 -p "Describe this picture."
```

`.jpg` and `.png` both work.

### Transcribe speech

```
curl -L -o asr.gguf https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf
```

```
curl -L -o asr-mmproj.gguf https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf
```

```
llama-mtmd-cli.exe -m asr.gguf --mmproj asr-mmproj.gguf --audio speech.mp3 -ngl 99 -p "Transcribe this audio."
```

`.mp3`, `.wav` and `.flac` all work. `--image` and `--audio` are in fact the same option, so
either name accepts either kind of file.

| file | bytes | SHA-256 |
|---|---:|---|
| `SmolVLM-500M-Instruct-Q8_0.gguf` | 436806912 | `9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4` |
| `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` | 108783360 | `d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150` |
| `Qwen3-ASR-0.6B-Q8_0.gguf` | 804749248 | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` |
| `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | 214392480 | `41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d` |

Speech recognition has been run on one GPU only so far, and 21 matrix operations in the audio
encoder are not yet implemented in this backend, so they fall back to the processor. It works,
but it is slower than it should be. `GGML_D3D12_STATS=1` lists exactly which operations fell back.

---

## 8. If something goes wrong

**"The code execution cannot proceed because dxcompiler.dll was not found"**
You moved the exe out of the folder. All six files have to stay together.

**No device listed by `--list-devices`**
Update your graphics driver. If it still shows nothing, your GPU or driver does not do D3D12
feature level 11_0.

**It runs but is slower than the processor**
Check you passed `-ngl 99`. Also check `--list-devices` is showing your real GPU and not a
software adapter called "Microsoft Basic Render Driver".

**A test fails, or the program stops with "device removed"**
That is a genuine bug report. Set this first and run it again, so the crash names the shader that
caused it instead of only printing an error code:

```
set GGML_D3D12_DRED=1
```

In PowerShell that is `$env:GGML_D3D12_DRED=1` instead.

Then open an issue at https://github.com/koza-coder/llama.cpp/issues with your GPU model, your
driver version, the exact command you ran, and the output.

### Useful switches

All of these are environment variables, set the same way as above. Every one is off by default.

| variable | what it does |
|---|---|
| `GGML_D3D12_DRED=1` | on a device removal, report which shader was running |
| `GGML_D3D12_STATS=1` | print dispatch counts, timings, and any operation that fell back to the processor |
| `GGML_D3D12_PROFILE=1` | per-shader timing |
| `GGML_D3D12_DISABLE=1` | hide the GPU entirely, so you can compare against the processor |

---

## 9. What has actually been tested

Honesty matters more than a long list here.

| GPU | result |
|---|---|
| AMD Radeon AI PRO R9700 (RDNA4) | all operation tests pass, models generate correctly |
| Moore Threads MTT S80 | all operation tests pass, models generate correctly |

Both on Windows 10 22H2. **No NVIDIA or Intel GPU has been tested at all.** If you run this on
one, the result is genuinely useful information either way - please report it.

Not implemented: the NVFP4, TQ1_0, TQ2_0, Q1_0 and Q2_0 quantisation formats. A model using one
of those will fall back to the processor for those tensors.

---

## Licence

llama.cpp is MIT. The DirectX Shader Compiler DLLs are Microsoft's, under the licences included
in the zip.
