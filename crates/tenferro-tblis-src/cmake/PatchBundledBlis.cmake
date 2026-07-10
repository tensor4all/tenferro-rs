set(TBLIS_CMAKE "${TBLIS_SOURCE_DIR}/CMakeLists.txt")
file(READ "${TBLIS_CMAKE}" TBLIS_CMAKE_CONTENTS)

set(BLIS_DISCOVERY_BLOCK "find_package(BLIS QUIET)

if(BLIS_FOUND)
    set(BLIS_TARGET BLIS::BLIS)
    message(CHECK_PASS \"found via cmake\")
    #not sure what to do here
    #set(BLIS_BINARY_DIR ...)
    add_custom_target(blis-install)
else()
    find_package(PkgConfig)
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
    pkg_check_modules(BLIS IMPORTED_TARGET blis>=2.0)
endif()")

set(BUNDLED_BLIS_BLOCK "set(BLIS_FOUND FALSE)")

string(REPLACE
    "${BLIS_DISCOVERY_BLOCK}"
    "${BUNDLED_BLIS_BLOCK}"
    TBLIS_PATCHED_CMAKE_CONTENTS
    "${TBLIS_CMAKE_CONTENTS}"
)

if(TBLIS_PATCHED_CMAKE_CONTENTS STREQUAL TBLIS_CMAKE_CONTENTS
   AND NOT TBLIS_CMAKE_CONTENTS MATCHES "set\\(BLIS_FOUND FALSE\\)")
    message(FATAL_ERROR "failed to patch TBLIS BLIS discovery block")
endif()

file(WRITE "${TBLIS_CMAKE}" "${TBLIS_PATCHED_CMAKE_CONTENTS}")
