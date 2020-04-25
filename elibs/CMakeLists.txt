#externalproject_add(tinyobjloader
#    PREFIX              ${CMAKE_SOURCE_DIR}/elibs/tinyobjloader
#    GIT_REPOSITORY      https://github.com/ipa-mah/tinyobjloader
#    GIT_TAG             master
#    UPDATE_COMMAND      ""
#    SOURCE_DIR          ${CMAKE_SOURCE_DIR}/elibs/tinyobjloader
#    CONFIGURE_COMMAND   ""
#    BUILD_COMMAND       ""
#    INSTALL_COMMAND     ""
#    )
add_library(tinyobjloader
  ${CMAKE_SOURCE_DIR}/elibs/tinyobjloader/tiny_obj_loader.cc
  ${CMAKE_SOURCE_DIR}/elibs/tinyobjloader/tiny_obj_loader.h)
target_include_directories(tinyobjloader
  PUBLIC
  tinyobjloader)
#add_library(tinyobjloader::tinyobjloader ALIAS tinyobjloader)
