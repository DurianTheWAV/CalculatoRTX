# CalculatoRTX - intègre un fichier binaire/texte (PTX OptiX) dans un fichier source C++.
# Utilisation : cmake -DINPUT=<fichier> -DOUTPUT=<fichier.cpp> -DSYMBOL=<nom> -P EmbedFile.cmake
if(NOT INPUT OR NOT OUTPUT OR NOT SYMBOL)
    message(FATAL_ERROR "EmbedFile.cmake : INPUT, OUTPUT et SYMBOL sont requis")
endif()

# INPUT peut être une liste (sortie de $<TARGET_OBJECTS:...>) : on prend le premier .ptx
set(_input "")
foreach(_f IN LISTS INPUT)
    if(_f MATCHES "\\.(ptx|optixir)$")
        set(_input "${_f}")
        break()
    endif()
endforeach()
if(NOT _input)
    list(GET INPUT 0 _input)
endif()

file(READ "${_input}" _hex HEX)
string(LENGTH "${_hex}" _hexlen)
math(EXPR _size "${_hexlen} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _bytes "${_hex}")
# retours à la ligne réguliers pour garder un fichier lisible par le compilateur
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" _bytes "${_bytes}")

file(WRITE "${OUTPUT}"
"// Fichier généré automatiquement par cmake/EmbedFile.cmake - ne pas modifier.\n"
"#include <cstddef>\n"
"extern \"C\" const unsigned char ${SYMBOL}[] = {\n${_bytes}\n0x00};\n"
"extern \"C\" const std::size_t ${SYMBOL}Size = ${_size};\n")
