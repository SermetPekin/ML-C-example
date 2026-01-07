#include "utils/base.h"

#ifdef _WIN32
    #include <windows.h>

    u32 plat_get_pagesize(void) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (u32)si.dwPageSize;
    }

    void* plat_mem_reserve(u64 size) {
        return VirtualAlloc(NULL, (SIZE_T)size, MEM_RESERVE, PAGE_NOACCESS);
    }

    b32 plat_mem_commit(void* ptr, u64 size) {
        return VirtualAlloc(ptr, (SIZE_T)size, MEM_COMMIT, PAGE_READWRITE) != NULL;
    }

    b32 plat_mem_decommit(void* ptr, u64 size) {
        return VirtualFree(ptr, (SIZE_T)size, MEM_DECOMMIT);
    }

    b32 plat_mem_release(void* ptr, u64 size) {
        (void)size;
        return VirtualFree(ptr, 0, MEM_RELEASE);
    }
#else
    #include <unistd.h>
    #include <sys/mman.h>

    #ifndef MAP_ANONYMOUS
    #define MAP_ANONYMOUS MAP_ANON
    #endif

    u32 plat_get_pagesize(void) {
        return (u32)sysconf(_SC_PAGESIZE);
    }

    void* plat_mem_reserve(u64 size) {
        void* ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return NULL;
        }
        return ptr;
    }

    b32 plat_mem_commit(void* ptr, u64 size) {
        return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
    }

    b32 plat_mem_decommit(void* ptr, u64 size) {
        madvise(ptr, size, MADV_DONTNEED);
        return mprotect(ptr, size, PROT_NONE) == 0;
    }

    b32 plat_mem_release(void* ptr, u64 size) {
        return munmap(ptr, size) == 0;
    }
#endif
