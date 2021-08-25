#pragma once

#ifdef _MSC_BUILD

#ifdef BUILD_DLL
#define __DLLEXPORT __declspec(dllexport)
#else
#define __DLLEXPORT __declspec(dllimport)
#endif

#else

#define __DLLEXPORT __attribute__ ((visibility ("default")))

#endif
