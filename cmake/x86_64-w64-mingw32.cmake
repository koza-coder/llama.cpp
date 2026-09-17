# Cross-compile for 64-bit Windows from Linux with the mingw-w64 GCC toolchain (posix thread model)
set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER   x86_64-w64-mingw32-gcc-posix )
set( CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++-posix )
set( CMAKE_RC_COMPILER  x86_64-w64-mingw32-windres )

set( CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32 )
set( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
set( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )
set( CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY )

# the GCC runtime stays dynamic (static linking duplicates symbols across the ggml DLLs):
# copy libstdc++-6.dll, libgcc_s_seh-1.dll, libgomp-1.dll and libwinpthread-1.dll next to the binaries
