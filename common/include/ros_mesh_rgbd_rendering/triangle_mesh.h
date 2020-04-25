#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tinyobjloader/tiny_obj_loader.h"

class MeshBase {
public:
  using Ptr = std::shared_ptr<MeshBase>;
  using ConstPtr = std::shared_ptr<const MeshBase>;
  MeshBase() {}

  MeshBase &operator+=(const MeshBase &mesh);
  MeshBase operator+(const MeshBase &mesh) const;

  /// Returns `True` if the mesh contains vertices.
  bool hasVertices() const { return vertices_.size() > 0; }

  /// Returns `True` if the mesh contains vertex normals.
  bool hasVertexNormals() const {
    return vertices_.size() > 0 && vertex_normals_.size() == vertices_.size();
  }

  /// Returns `True` if the mesh contains vertex colors.
  bool hasVertexColors() const {
    return vertices_.size() > 0 && vertex_colors_.size() == vertices_.size();
  }

  /// Normalize vertex normals to length 1.
  MeshBase &normalizeNormals() {
    for (size_t i = 0; i < vertex_normals_.size(); i++) {
      vertex_normals_[i].normalize();
      if (std::isnan(vertex_normals_[i](0))) {
        vertex_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
      }
    }
    return *this;
  }
  MeshBase &clear();

public:
  /// Vertex coordinates.
  std::vector<Eigen::Vector3d> vertices_;
  /// Vertex normals.
  std::vector<Eigen::Vector3d> vertex_normals_;
  /// RGB colors of vertices.
  std::vector<Eigen::Vector3d> vertex_colors_;
};

MeshBase &MeshBase::clear() {
  vertices_.clear();
  vertex_normals_.clear();
  vertex_colors_.clear();
  return *this;
}

class TriangleMesh : public MeshBase {
public:
  using Ptr = std::shared_ptr<TriangleMesh>;
  using ConstPtr = std::shared_ptr<const TriangleMesh>;
  /// \brief Default Constructor.
  TriangleMesh() : MeshBase() {}

  TriangleMesh &operator+=(const TriangleMesh &mesh);
  TriangleMesh operator+(const TriangleMesh &mesh) const;
  /// Returns `true` if the mesh contains triangles.
  bool hasTriangles() const {
    return vertices_.size() > 0 && triangles_.size() > 0;
  }

  /// Returns `true` if the mesh contains triangle normals.
  bool hasTriangleNormals() const {
    return hasTriangles() && triangles_.size() == triangle_normals_.size();
  }

  /// Returns `true` if the mesh contains adjacency normals.
  bool hasAdjacencyList() const {
    return vertices_.size() > 0 && adjacency_list_.size() == vertices_.size();
  }

  bool hasTriangleUvs() const {
    return hasTriangles() && triangle_uvs_.size() == 3 * triangles_.size();
  }

  /// Returns `true` if the mesh has texture.
  bool hasTextures() const {
    bool is_all_texture_valid = std::accumulate(
        textures_.begin(), textures_.end(), true,
        [](bool a, const cv::Mat &b) { return a && !b.empty(); });
    return !textures_.empty() && is_all_texture_valid;
  }

  bool hasTriangleMaterialIds() const {
    return hasTriangles() && triangle_material_ids_.size() == triangles_.size();
  }

  /// Normalize both triangle normals and vertex normals to length 1.
  TriangleMesh &NormalizeNormals() {
    MeshBase::normalizeNormals();
    for (size_t i = 0; i < triangle_normals_.size(); i++) {
      triangle_normals_[i].normalize();
      if (std::isnan(triangle_normals_[i](0))) {
        triangle_normals_[i] = Eigen::Vector3d(0.0, 0.0, 1.0);
      }
    }
    return *this;
  }

  TriangleMesh &clear();

public:
  /// List of triangles denoted by the index of points forming the triangle.
  std::vector<Eigen::Vector3i> triangles_;
  /// Triangle normals.
  std::vector<Eigen::Vector3d> triangle_normals_;
  /// The set adjacency_list[i] contains the indices of adjacent vertices of
  /// vertex i.
  std::vector<std::unordered_set<int>> adjacency_list_;
  /// List of uv coordinates per triangle.
  std::vector<Eigen::Vector2d> triangle_uvs_;
  /// List of material ids.
  std::vector<int> triangle_material_ids_;
  /// Textures of the image.
  std::vector<cv::Mat> textures_;
};

TriangleMesh &TriangleMesh::clear() {
  MeshBase::clear();
  triangles_.clear();
  triangle_normals_.clear();
  adjacency_list_.clear();
  triangle_uvs_.clear();
  triangle_material_ids_.clear();
  textures_.clear();
  return *this;
}

static bool readTextureMeshfromOBJFile(const std::string filename,
                                       TriangleMesh::Ptr &mesh) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  std::string mtl_base_path;

  size_t slash_pos = filename.find_last_of("/\\");
  if (slash_pos == std::string::npos) {
    mtl_base_path = "";
  } else {
    mtl_base_path = filename.substr(0, slash_pos + 1);
  }

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename.c_str(), mtl_base_path.c_str());
  if (!warn.empty()) {
    std::cout << "Read OBJ failed: " << warn << std::endl;
    ;
  }
  if (!err.empty()) {
    std::cout << "Read OBJ failed: " << err << std::endl;
  }

  if (!ret) {
    return false;
  }

  mesh->clear();

  // copy vertex and data
  for (size_t vidx = 0; vidx < attrib.vertices.size(); vidx += 3) {
    tinyobj::real_t vx = attrib.vertices[vidx + 0];
    tinyobj::real_t vy = attrib.vertices[vidx + 1];
    tinyobj::real_t vz = attrib.vertices[vidx + 2];
    mesh->vertices_.push_back(Eigen::Vector3d(vx, vy, vz));
  }

  for (size_t vidx = 0; vidx < attrib.colors.size(); vidx += 3) {
    tinyobj::real_t r = attrib.colors[vidx + 0];
    tinyobj::real_t g = attrib.colors[vidx + 1];
    tinyobj::real_t b = attrib.colors[vidx + 2];
    mesh->vertex_colors_.push_back(Eigen::Vector3d(r, g, b));
  }

  // resize normal data and create bool indicator vector
  mesh->vertex_normals_.resize(mesh->vertices_.size());
  std::vector<bool> normals_indicator(mesh->vertices_.size(), false);

  // copy face data and copy normals data
  // append face-wise uv data
  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];
      if (fv != 3) {
        printf("Read OBJ failed: facet with number of vertices not "
               "equal to 3");
        return false;
      }
      Eigen::Vector3i facet;
      for (int v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        int vidx = idx.vertex_index;
        facet(v) = vidx;

        if (!attrib.normals.empty() && !normals_indicator[vidx] &&
            (3 * idx.normal_index + 2) < int(attrib.normals.size())) {
          tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
          tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
          tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
          mesh->vertex_normals_[vidx](0) = nx;
          mesh->vertex_normals_[vidx](1) = ny;
          mesh->vertex_normals_[vidx](2) = nz;
          normals_indicator[vidx] = true;
        }

        if (!attrib.texcoords.empty() &&
            2 * idx.texcoord_index + 1 < int(attrib.texcoords.size())) {
          tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
          tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
          mesh->triangle_uvs_.push_back(Eigen::Vector2d(tx, ty));
        }
      }
      mesh->triangles_.push_back(facet);
      mesh->triangle_material_ids_.push_back(shapes[s].mesh.material_ids[f]);
      index_offset += fv;
    }
  }

  // if not all normals have been set, then remove the vertex normals
  bool all_normals_set =
      std::accumulate(normals_indicator.begin(), normals_indicator.end(), true,
                      [](bool a, bool b) { return a && b; });
  if (!all_normals_set) {
    mesh->vertex_normals_.clear();
  }

  // if not all triangles have corresponding uvs, then remove uvs
  if (3 * mesh->triangles_.size() != mesh->triangle_uvs_.size()) {
    mesh->triangle_uvs_.clear();
  }

  // Now we assert only one shape is stored, we only select the first
  // diffuse material
  for (auto &material : materials) {
    if (!material.diffuse_texname.empty()) {
      mesh->textures_.push_back(
          cv::imread(mtl_base_path + material.diffuse_texname));
    }
  }

  return true;
}

#endif // TRIANGLE_MESH_H
