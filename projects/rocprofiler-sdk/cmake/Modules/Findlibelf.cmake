# Try to find libelf headers and libraries.
#
# Usage of this module as follows:
#
# find_package(libelf)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
# libelf_ROOT         Set this variable to the root installation of
#                     libelf if the module has problems finding the
#                     proper installation path.
#
# Variables defined by this module:
#
# libelf_FOUND              System has libelf libraries and headers
# libelf_LIBRARIES          The libelf library
# libelf_INCLUDE_DIRS       The location of libelf headers
#
# Interface targets defined by this module:
#
# libelf::libelf
#

find_package(PkgConfig)

set(_libelf_ROOT_HINTS
    ${libelf_ROOT}
    ${rocm_version_DIR}
    ${ROCM_PATH}
    $ENV{ROCM_PATH}
    ${rocm_version_DIR}/lib/rocm_sysdeps
    ${ROCM_PATH}/lib/rocm_sysdeps
    $ENV{ROCM_PATH}/lib/rocm_sysdeps)

if(PkgConfig_FOUND)
    set(ENV{PKG_CONFIG_SYSTEM_INCLUDE_PATH} "")
    pkg_check_modules(ELF libelf)

    if(ELF_FOUND
       AND ELF_INCLUDE_DIRS
       AND ELF_LIBRARIES)
        set(libelf_INCLUDE_DIR
            "${ELF_INCLUDE_DIRS}"
            CACHE FILEPATH "libelf include directory")
        set(libelf_LIBRARY
            "${ELF_LIBRARIES}"
            CACHE FILEPATH "libelf libraries")
    endif()
endif()

if(NOT libelf_INCLUDE_DIR OR NOT libelf_LIBRARY)
    find_path(
        libelf_ROOT_DIR
        NAMES include/elf.h
        HINTS ${_libelf_ROOT_HINTS}
        PATHS ${_libelf_ROOT_HINTS})

    mark_as_advanced(libelf_ROOT_DIR)

    find_path(
        libelf_INCLUDE_DIR
        NAMES elf.h
        HINTS ${_libelf_ROOT_HINTS}
        PATHS ${_libelf_ROOT_HINTS}
        PATH_SUFFIXES include)

    find_library(
        libelf_LIBRARY
        NAMES elf
        HINTS ${_libelf_ROOT_HINTS}
        PATHS ${_libelf_ROOT_HINTS}
        PATH_SUFFIXES lib lib64 lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libelf DEFAULT_MSG libelf_LIBRARY libelf_INCLUDE_DIR)

if(libelf_FOUND)
    if(NOT TARGET libelf::libelf)
        add_library(libelf::libelf INTERFACE IMPORTED)
    endif()

    if(TARGET PkgConfig::ELF AND ELF_FOUND)
        target_link_libraries(libelf::libelf INTERFACE PkgConfig::ELF)
    else()
        target_link_libraries(libelf::libelf INTERFACE ${libelf_LIBRARY})
        target_include_directories(libelf::libelf SYSTEM INTERFACE ${libelf_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(libelf_INCLUDE_DIR libelf_LIBRARY)
unset(_libelf_ROOT_HINTS)
