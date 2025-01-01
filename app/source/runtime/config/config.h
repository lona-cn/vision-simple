#pragma once

#if defined _WIN32 || defined __CYGWIN__
    #ifdef _MSC_VER // MSVC
        #ifdef EXPORTING_VISION_SIMPLE
            #define VISION_SIMPLE_API __declspec(dllexport)
        #else
            #define VISION_SIMPLE_API __declspec(dllimport)
        #endif
    #else // GCC 或其他 Windows 编译器
        #ifdef EXPORTING_VISION_SIMPLE
            #define VISION_SIMPLE_API __attribute__((dllexport))
        #else
            #define VISION_SIMPLE_API __attribute__((dllimport))
        #endif
    #endif
#else // 非 Windows 平台（Linux/macOS）
    #if __GNUC__ >= 4
        #define VISION_SIMPLE_API __attribute__((visibility("default")))
    #else
        #define VISION_SIMPLE_API
    #endif
#endif