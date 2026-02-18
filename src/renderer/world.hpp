#pragma once
#include <math>
#include <draw>
#include <rt>
#include <limits>
#include <type_traits>
#include <variant>
#include <utility>
#include <concepts>
#include <vector>
#include <span>
#include <ranges>

#ifdef ENABLE_VULKAN_SUPPORT
#include <vulkan/vulkan.hpp>
#endif

namespace renderer {
    /// A simple floating point color type.
    /// It implements a lossy implicit conversion to and from draw::Color.
    /// I do not particularly care as I would never touch floats (only integers or fixed point) in my own code.
    /// However, since the assignment said that doing things properly would not count, here's a horrible, imprecise
    /// color type with all the misguided semantics like NaN and infinity.
    struct Color final {
        f32 r { 0.f }, g { 0.f }, b { 0.f };

        constexpr explicit(false) Color(draw::Color color = draw::color::CLEAR)
            : r(f32(color.r) / 255), g(f32(color.g) / 255), b(f32(color.b) / 255) {}

        constexpr explicit(false) operator draw::Color () const {
            return draw::Color::rgba(
                std::clamp(r * 255.f, 0.f, 255.f),
                std::clamp(g * 255.f, 0.f, 255.f),
                std::clamp(b * 255.f, 0.f, 255.f)
            );
        }

        constexpr explicit(false) Color(math::Vector<f32, 3> vector) : r(vector[0]), g(vector[1]), b(vector[2]) {}

        constexpr explicit(false) operator math::Vector<f32, 3> () const {
            return { r, g, b };
        }

        constexpr Color(f32 r, f32 g, f32 b) : r(r), g(g), b(b) {}

        [[clang::always_inline]] [[gnu::const]]
        static constexpr Color rgb(f32 r, f32 g, f32 b) noexcept {
            return Color { r, g, b };
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto operator==(this Color self, Color other) noexcept -> bool {
            return self.r == other.r and self.g == other.g and self.b == other.b;
        }

        [[clang::always_inline]] [[gnu::const]]
        constexpr auto operator!=(this Color self, Color other) noexcept -> bool {
            return !(self == other);
        }
    };

    struct Hit final {
		math::Vector<f32, 3> origin;
		math::Vector<f32, 3> normal;
		f32 distance { std::numeric_limits<f32>::max() };

		usize material_index { 0 };
	};

    struct Sphere final {
        math::Vector<f32, 3> position;
        f32 radius;
    };

    struct Plane final {
        math::Vector<f32, 3> position, normal;
    };

    struct Mesh final {
        struct FaceElement final {
            usize vertex;
            usize texture_coordinate;
            usize normal;
        };

        using Face = std::array<FaceElement, 3>;

        math::Vector<f32, 3> position;
        std::vector<math::Vector<f32, 3>> vertices;
        std::vector<math::Vector<f32, 3>> texture_coordinates;
        std::vector<math::Vector<f32, 3>> normals;
        std::vector<Face> faces;
        f32 scale { 1.f };
        math::Angle<f32> pitch { 0.f };
        math::Angle<f32> yaw { 0.f };
        math::Angle<f32> roll { 0.f };

        std::optional<draw::Image> diffuse;
        std::optional<draw::Image> gloss;
        std::optional<draw::Image> normal;
        std::optional<draw::Image> specular;

        enum class Shading {
            Flat,
            Smooth
        } shading { Shading::Flat };

        struct BvhNode final {
            math::Vector<f32, 3> bound_min;
            math::Vector<f32, 3> bound_max;

            usize face_index;
            usize face_count;

            Box<BvhNode> left;
            Box<BvhNode> right;
        };

        Box<BvhNode> bvh;

        Mesh() = default;

        Mesh(Mesh&&) = default;
        Mesh& operator=(Mesh&&) = default;

        Mesh(Mesh const&) = delete;
        Mesh& operator=(Mesh const&) = delete;

        auto clone() const -> Mesh {
            Mesh ret;

            ret.position = position;
            ret.vertices = vertices;
            ret.texture_coordinates = texture_coordinates;
            ret.normals = normals;
            ret.faces = faces;
            ret.scale = scale;
            ret.pitch = pitch;
            ret.yaw = yaw;
            ret.roll = roll;
            ret.shading = shading;

            if (bvh) ret.compute_bvh();

            return ret;
        }

      private:
        static void compute_bounds(
            std::span<const math::Vector<f32, 3>> vertices,
            std::span<const Face> faces,
            math::Vector<f32, 3>& out_min,
            math::Vector<f32, 3>& out_max
        ) {
            using Vector = math::Vector<f32, 3>;
            // Silly name because the preprocessor exists and defines INFINITY. I hate C++.
            constexpr static f32 INF = std::numeric_limits<f32>::infinity();

            out_min = {  INF,  INF,  INF };
            out_max = { -INF, -INF, -INF };

            for (auto const& face : faces) {
                for (auto [vertex, _, _] : face) {
                    const auto& v = vertices[vertex];
                    for (i32 a = 0; a < 3; a += 1) {
                        out_min[a] = std::min(out_min[a], v[a]);
                        out_max[a] = std::max(out_max[a], v[a]);
                    }
                }
            }
        }

        static auto partition_faces(
            std::span<const math::Vector<f32, 3>> vertices,
            std::span<Face> faces,
            i32 axis,
            f32 split
        ) -> usize {
            usize i = 0;
            usize j = faces.size();

            while (i < j) {
                const auto& f = faces[i];
                math::Vector<f32, 3> c = (vertices[f[0].vertex] + vertices[f[1].vertex] + vertices[f[2].vertex]) / 3.f;

                if (c[axis] < split) {
                    i += 1;
                } else {
                    j -= 1;
                    std::swap(faces[i], faces[j]);
                }
            }

            return i;
        }

        static auto build_bvh(
            std::span<const math::Vector<f32, 3>> vertices,
            std::span<Face> faces,
            usize face_offset,
            usize leaf_size = 4
        ) -> Box<Mesh::BvhNode> {
            using Node = Mesh::BvhNode;
            auto node = Box<Node>::make();

            // Compute bounding box
            compute_bounds(vertices, faces, node->bound_min, node->bound_max);

            node->face_index = face_offset;
            node->face_count = faces.size();

            if (faces.size() <= leaf_size) return node;

            // Choose axis with largest extent
            math::Vector<f32, 3> extent = node->bound_max - node->bound_min;
            i32 axis = 0;
            if (extent[1] > extent[axis]) axis = 1;
            if (extent[2] > extent[axis]) axis = 2;

            f32 split = (node->bound_min[axis] + node->bound_max[axis]) * 0.5f;

            // Partition in-place
            usize mid = partition_faces(vertices, faces, axis, split);

            // If partition fails (all on one side), make leaf
            if (mid == 0 or mid == faces.size()) return node;

            auto left_span  = faces.first(mid);
            auto right_span = faces.last(faces.size() - mid);

            node->left = build_bvh(vertices, left_span, face_offset, leaf_size);
            node->right = build_bvh(vertices, right_span, face_offset + mid, leaf_size);

            return node;
        }

      public:
        void compute_bvh() {
            if (faces.empty()) {
                bvh = Box<BvhNode>();
                return;
            }
            bvh = build_bvh(vertices, faces, 0);
        }

        static auto intersect_aabb(
            math::Vector<f32, 3> const& origin,
            math::Vector<f32, 3> const& dir_inv,
            math::Vector<f32, 3> const& bmin,
            math::Vector<f32, 3> const& bmax,
            f32& tmin_out,
            f32& tmax_out
        ) -> bool {
            f32 tmin = -std::numeric_limits<f32>::infinity();
            f32 tmax =  std::numeric_limits<f32>::infinity();

            for (i32 i = 0; i < 3; i += 1) {
                f32 t0 = (bmin[i] - origin[i]) * dir_inv[i];
                f32 t1 = (bmax[i] - origin[i]) * dir_inv[i];
                if (t0 > t1) std::swap(t0, t1);

                tmin = std::max(tmin, t0);
                tmax = std::min(tmax, t1);

                if (tmax < tmin) return false;
            }

            tmin_out = tmin;
            tmax_out = tmax;
            return true;
        }

        static auto intersect_triangle(
            math::Vector<f32, 3> const& origin,
            math::Vector<f32, 3> const& dir,
            math::Vector<f32, 3> const& v0,
            math::Vector<f32, 3> const& v1,
            math::Vector<f32, 3> const& v2,
            math::Vector<f32, 3> const& n0,
            math::Vector<f32, 3> const& n1,
            math::Vector<f32, 3> const& n2,
            Shading shading
        ) -> std::optional<Hit> {
            // Möller–Trumbore
            constexpr f32 EPS = 1e-6f;
            auto e1 = v1 - v0;
            auto e2 = v2 - v0;

            auto pvec = dir.cross(e2);
            f32 det = e1.dot(pvec);
            if (std::abs(det) < EPS) return std::nullopt;
            f32 inv_det = 1.0f / det;

            auto tvec = origin - v0;
            f32 u = tvec.dot(pvec) * inv_det;
            if (u < 0 || u > 1) return std::nullopt;

            auto qvec = tvec.cross(e1);
            f32 v = dir.dot(qvec) * inv_det;
            if (v < 0 || u + v > 1) return std::nullopt;

            f32 t = e2.dot(qvec) * inv_det;
            if (t < EPS) return std::nullopt;

            Hit hit;
            hit.origin = origin + dir * t;
            hit.normal = shading == Shading::Flat
                ? e1.cross(e2).normalized()
                : (n0 * (1.f - u - v) + n1 * u + n2 * v).normalized();
            hit.distance = t;
            return hit;
        }

        bool intersect_bvh(
            BvhNode const* node,
            math::Vector<f32, 3> const& origin,
            math::Vector<f32, 3> const& dir,
            math::Vector<f32, 3> const& dir_inv,
            f32& best_distance,
            Hit& best_hit
        ) const {
            f32 tmin, tmax;
            if (not intersect_aabb(origin, dir_inv, node->bound_min, node->bound_max, tmin, tmax))
                return false;

            bool hit_any = false;

            // Leaf node
            if (not node->left and not node->right) {
                for (usize i = 0; i < node->face_count; i += 1) {
                    auto const& face = faces[node->face_index + i];
                    auto const& v0 = vertices[face[0].vertex];
                    auto const& v1 = vertices[face[1].vertex];
                    auto const& v2 = vertices[face[2].vertex];
                    auto const& n0 = normals[face[0].normal];
                    auto const& n1 = normals[face[1].normal];
                    auto const& n2 = normals[face[2].normal];

                    if (auto hit = intersect_triangle(origin, dir, v0, v1, v2, n0, n1, n2, shading)) {
                        if (hit->distance < best_distance) {
                            best_distance = hit->distance;
                            best_hit = *hit;
                            hit_any = true;
                        }
                    }
                }
            } else {
                if (node->left)
                    hit_any |= intersect_bvh(node->left.raw(), origin, dir, dir_inv, best_distance, best_hit);
                if (node->right)
                    hit_any |= intersect_bvh(node->right.raw(), origin, dir, dir_inv, best_distance, best_hit);
            }

            return hit_any;
        }

        [[gnu::const]]
        math::Matrix<f32, 4, 4> local_to_world() const {
            using Matrix = math::Matrix<f32, 4, 4>;

            return Matrix::scaling(scale, scale, scale)
                 * Matrix::rotation(Matrix::RotationAxis::Pitch, pitch)
                 * Matrix::rotation(Matrix::RotationAxis::Yaw, yaw)
                 * Matrix::rotation(Matrix::RotationAxis::Roll, roll)
                 * Matrix::translation(position.x(), position.y(), position.z());
        }

        [[gnu::const]]
        math::Matrix<f32, 4, 4> world_to_local() const {
            return local_to_world().inverse();
        }

        auto intersect(math::Vector<f32, 3> origin, math::Vector<f32, 3> direction) const -> std::optional<Hit> {
            if (not bvh) return std::nullopt;

            auto world_to_local_mat = world_to_local();
            math::Vector<f32, 4> o4 { origin,    1.f };
            math::Vector<f32, 4> d4 { direction, 0.f };

            auto local_origin = (o4 * world_to_local_mat);
            auto local_dir    = (d4 * world_to_local_mat).normalized();
            auto local_dir_inv = math::Vector<f32, 3>{
                1.f / local_dir[0],
                1.f / local_dir[1],
                1.f / local_dir[2]
            };

            f32 best_distance = std::numeric_limits<f32>::max();
            Hit best_hit;

            if (intersect_bvh(bvh.raw(), local_origin, local_dir, local_dir_inv, best_distance, best_hit)) {
                // Convert hit point and normal back to world space
                auto local_to_world_mat = local_to_world();
                math::Vector<f32, 4> hit4 { best_hit.origin[0], best_hit.origin[1], best_hit.origin[2], 1.0f };
                math::Vector<f32, 4> normal4 { best_hit.normal[0], best_hit.normal[1], best_hit.normal[2], 0.0f };

                best_hit.origin = (hit4 * local_to_world_mat);
                best_hit.normal = (normal4 * local_to_world_mat).normalized();
                best_hit.distance = (best_hit.origin - origin).magnitude(); // length?

                return best_hit;
            }

            return std::nullopt;
        }
    };

    struct PointLight final {
        math::Vector<f32, 3> position;
        renderer::Color color;
    };

	class World;

	class Material {
	  public:
    	virtual auto shade(Hit hit, World const& world, u32 depth) const -> renderer::Color = 0;

        virtual constexpr auto operator==(Material const& rhs) const -> bool = 0;

        virtual ~Material() {}
	};

    class SolidColorMaterial final : public Material {
        renderer::Color color;

      public:
		constexpr explicit SolidColorMaterial(renderer::Color color) : color(color) {}

		auto shade(Hit hit, World const& world, u32 depth) const -> renderer::Color override {
		    return color;
		}

		constexpr auto operator==(Material const& rhs) const -> bool override {
		    const auto same = dynamic_cast<SolidColorMaterial const*>(&rhs);
		    return same and color == same->color;
		}
	};

	class LambertMaterial final : public Material {
	    renderer::Color color;
		f32 diffuse_reflectance;

	  public:
		constexpr explicit LambertMaterial(renderer::Color color, f32 diffuse_reflectance = 1.f)
		    : color(color), diffuse_reflectance(diffuse_reflectance) {}

		auto shade(Hit hit, World const& world, u32 depth) const -> renderer::Color override;

		constexpr auto operator==(Material const& rhs) const -> bool override {
		    const auto same = dynamic_cast<LambertMaterial const*>(&rhs);
		    return same and color == same->color;
		}
	};

	class BsdfMaterial final : public Material {
	    renderer::Color color;
		renderer::Color emissive;
		f32 roughness;
		f32 metallic;

	  public:
		struct Config final {
		    renderer::Color color { draw::color::BLACK };
			renderer::Color emissive { draw::color::BLACK };
			f32 roughness { 1.f };
			f32 metallic { 0.f };
		};

		constexpr explicit(false) BsdfMaterial(Config config)
		    : color(config.color), emissive(config.emissive), roughness(config.roughness), metallic(config.metallic) {}

		auto shade(Hit hit, World const& world, u32 depth) const -> renderer::Color override;

		constexpr auto operator==(Material const& rhs) const -> bool override {
		    const auto same = dynamic_cast<BsdfMaterial const*>(&rhs);
		    return same and color == same->color and roughness == same->roughness and metallic == same->metallic;
		}

		enum class Mode {
		    Default,
			Diffuse,
			CookTorrance,
			Fresnel,
			NormalDistribution,
			Microfacets
		};

		enum class GiMode {
            None, Simple
        };
	};

	constexpr std::ostream& operator<<(std::ostream& os, BsdfMaterial::Mode const& value) {
	    switch (value) {
            case BsdfMaterial::Mode::Default:            os << "Default";            break;
            case BsdfMaterial::Mode::Diffuse:            os << "Diffuse";            break;
            case BsdfMaterial::Mode::CookTorrance:       os << "CookTorrance";       break;
            case BsdfMaterial::Mode::Fresnel:            os << "Fresnel";            break;
            case BsdfMaterial::Mode::NormalDistribution: os << "NormalDistribution"; break;
            case BsdfMaterial::Mode::Microfacets:        os << "Microfacets";        break;
        }

        return os;
    }

    constexpr std::ostream& operator<<(std::ostream& os, BsdfMaterial::GiMode const& value) {
	    switch (value) {
            case BsdfMaterial::GiMode::None:   os << "None";   break;
            case BsdfMaterial::GiMode::Simple: os << "Simple"; break;
        }

        return os;
    }

    using Shape = std::variant<Sphere, Plane, Mesh>;

    template <typename T, typename... Us> concept any_of = (std::same_as<T, Us> or ...);

    template <typename T, typename U> concept subtype = std::is_base_of<U, T>::value;

    template <
        std::input_iterator I,
        // ???
        // std::equality_comparable_with<decltype(*std::declval<typename std::iterator_traits<I>::value_type>)> E
        typename E
    > auto indirect_find(I begin, I end, E const& e) -> I {
        for (; begin != end; ++begin) if (**begin == e) return begin;
        return begin;
    }

    static auto load_mesh(Io& io, std::string_view path) -> Mesh {
        const auto data = io.read_file(path);
        const auto obj = std::string_view((char const*) data.data(), data.size());

        Mesh mesh;

        for (const auto line : obj | std::views::split('\n')) {
            auto components = line | std::views::split(' ');

            auto it = components.begin();
            const auto next = [&] -> std::optional<std::string_view> {
                if (it == components.end()) [[unlikely]] return std::nullopt;
                else {
                    auto const& ret = *it++;
                    return std::string_view(&*ret.begin(), std::ranges::distance(ret));
                }
            };

            if (const auto id = next()) {
                if (id == "v") {
                    mesh.vertices.emplace_back(
                        std::stof(std::string(next().value())),
                        std::stof(std::string(next().value())),
                        std::stof(std::string(next().value()))
                    );
                } else if (id == "vt") {
                    mesh.texture_coordinates.emplace_back(
                        next().transform([] (auto e) { return std::stof(std::string(e)); }).value(),
                        next().transform([] (auto e) { return std::stof(std::string(e)); }).value_or(0.f),
                        next().transform([] (auto e) { return std::stof(std::string(e)); }).value_or(0.f)
                    );
                } else if (id == "vn") {
                    mesh.normals.emplace_back(
                        std::stof(std::string(next().value())),
                        std::stof(std::string(next().value())),
                        std::stof(std::string(next().value()))
                    );
                } else if (id == "f") {
                    const static auto parse_face_element = [&] () -> Mesh::FaceElement {
                        auto element_components = next().value() | std::views::split('/');

                        auto it = element_components.begin();
                        const auto next_element_component = [&] -> std::optional<std::string_view> {
                            if (it == element_components.end()) [[unlikely]] return std::nullopt;
                            else {
                                auto const& ret = *it++;
                                return std::string_view(&*ret.begin(), std::ranges::distance(ret));
                            }
                        };

                        // C++ is a sad language so we can't use the constructor as a callable.
                        // That being said it sucks in the first place because it requires allocating a string
                        // to be able to parse the number. It works with C strings but that's not a slice >:(
                        // Terrible design, all of this.
                        // There's also no flat transform which is neat, and the error handling is exceptions
                        // which is rather stupid for something so small and commonly used in loops.
                        constexpr auto sv_to_u64 = [] (auto&& v) -> u64 {
                            try {
                                return std::stoul(std::string(v));
                            } catch (std::exception e) {
                                return 0;
                            }
                        };

                        return {
                            .vertex             = next_element_component().transform(sv_to_u64).value() - 1,
                            .texture_coordinate = next_element_component().transform(sv_to_u64).value_or(0) - 1,
                            .normal             = next_element_component().transform(sv_to_u64).value_or(0) - 1
                        };
                    };

                    mesh.faces.push_back({
                        parse_face_element(),
                        parse_face_element(),
                        parse_face_element()
                    });
                } else if (id == "s") [[unlikely]] {
                    mesh.shading = std::stoul(std::string(next().value())) ? Mesh::Shading::Smooth : Mesh::Shading::Flat;
                }
            }
        }

        mesh.compute_bvh();

        return mesh;
    }

    enum class RasterMode {
        Default,
        Wireframe,
        DepthBuffer
    };

    constexpr std::ostream& operator<<(std::ostream& os, RasterMode const& value) {
	    switch (value) {
            case RasterMode::Default:     os << "Default";     break;
            case RasterMode::Wireframe:   os << "Wireframe";   break;
            case RasterMode::DepthBuffer: os << "DepthBuffer"; break;
        }

        return os;
    }

    class World final {
        // Collection of shapes and their bound materials.
        std::vector<std::pair<Shape, usize>> object_data;
        // Collection of materials where indices remain consistent.
        std::vector<Box<Material>> material_data;
        // Collection of point lights.
        std::vector<PointLight> light_data;

        math::Vector<f32, 3> camera_position;
        math::Angle<f32> camera_pitch { 0.f };
        math::Angle<f32> camera_yaw { 0.f };
        math::Angle<f32> camera_roll { 0.f };

        renderer::Color background_color { draw::color::BLACK };

        math::Angle<f32> fov { math::deg(80.f).radians() };
        bool checkerboard { true };
        bool shadows { true };
        bool normal_mapping { true };
        BsdfMaterial::Mode bsdf_mode { BsdfMaterial::Mode::Default };
        BsdfMaterial::GiMode gi_mode { BsdfMaterial::GiMode::None };

        RasterMode raster_mode { RasterMode::Default };

      public:
        World() {
            material_data.push_back(Box<SolidColorMaterial>::make(SolidColorMaterial(draw::color::pico::RED)));
        }

        World(World const&) = delete;
        World(World&&) = default;
        World& operator=(World const&) = delete;
        World& operator=(World&&) = default;

        auto lights() const -> std::span<const PointLight> {
          return light_data;
        }

        auto objects() const -> std::span<const std::pair<Shape, usize>> {
            return object_data;
        }

        auto material(usize index) const -> Material const& {
            return *material_data[index];
        }

        [[gnu::const]]
        auto get_background_color() const -> renderer::Color {
            return background_color;
        }

        auto set_background_color(renderer::Color color) {
            background_color = color;
        }

        auto get_normal_mapping() const -> bool {
            return normal_mapping;
        }

        void set_normal_mapping(bool value) {
            normal_mapping = value;
        }

        /// Since objects are not guaranteed a stable address in memory, a reference type is provided
        /// to provide a handle to an object which can outlive storage resizing.
        template <typename Object> class Ref {
            World * world;
            usize index;

            Ref(World * world, usize index) : world(world), index(index) {}

          public:
            friend class World;

            Ref() : world(nullptr), index(0) {}

            auto operator*() const -> Object& { return std::get<Object>(world->object_data[index].first); }
            auto operator->() const -> Object* { return &std::get<Object>(world->object_data[index].first); }

            operator bool () const { return world; }
        };

        template <any_of<Sphere, Plane, Mesh> Object, subtype<Material> Mat> auto add(Object object, Mat material) -> Ref<Object> {
            usize material_index;
            if (const auto it = indirect_find(material_data.begin(), material_data.end(), material); it != material_data.end()) {
                material_index = it - material_data.begin();
            } else {
                material_data.push_back(Box<Mat>::make(material));
                material_index = material_data.size() - 1;
            }

            object_data.emplace_back(std::move(object), material_index);
            return { this, object_data.size() - 1 };
        }

        template <any_of<Sphere, Plane, Mesh> Object> auto add(Object object) -> Ref<Object> {
            return add(object, SolidColorMaterial(draw::color::pico::RED));
        }

        void add(PointLight light) {
            light_data.push_back(light);
        }

        void move(math::Vector<f32, 3> vector) {
            using Matrix = math::Matrix<f32, 3, 3>;
            const auto rotation_matrix = Matrix::rotation(Matrix::RotationAxis::Yaw, camera_yaw);
            camera_position += vector * rotation_matrix;
        }

        void set_fov(math::Angle<f32> angle) {
            fov = angle;
        }

        auto get_fov() const -> math::Angle<f32> {
            return fov;
        }

        void set_checkerboard(bool value) {
            checkerboard = value;
        }

        auto get_checkerboard() const -> bool {
            return checkerboard;
        }

        void set_shadows(bool value) {
            shadows = value;
        }

        [[gnu::const]]
        auto get_shadows() const -> bool {
            return shadows;
        }

        void set_bsdf_mode(BsdfMaterial::Mode value) {
            bsdf_mode = value;
        }

        [[gnu::const]]
        auto get_bsdf_mode() const -> BsdfMaterial::Mode {
            return bsdf_mode;
        }

        void cycle_bsdf_mode() {
            using enum BsdfMaterial::Mode;
            switch (bsdf_mode) {
                case Default:            bsdf_mode = Diffuse;            break;
                case Diffuse:            bsdf_mode = CookTorrance;       break;
                case CookTorrance:       bsdf_mode = Fresnel;            break;
                case Fresnel:            bsdf_mode = NormalDistribution; break;
                case NormalDistribution: bsdf_mode = Microfacets;        break;
                case Microfacets:        bsdf_mode = Default;            break;
            }
        }

        void set_gi_mode(BsdfMaterial::GiMode value) {
            gi_mode = value;
        }

        [[gnu::const]]
        auto get_gi_mode() const -> BsdfMaterial::GiMode {
            return gi_mode;
        }

        void cycle_gi_mode() {
            using enum BsdfMaterial::GiMode;
            switch (gi_mode) {
                case None:   gi_mode = Simple; break;
                case Simple: gi_mode = None;   break;
            }
        }

        void cycle_raster_mode() {
            using enum RasterMode;
            switch (raster_mode) {
                case Default:     raster_mode = Wireframe;   break;
                case Wireframe:   raster_mode = DepthBuffer; break;
                case DepthBuffer: raster_mode = Default;     break;
            }
        }

        auto get_raster_mode() const {
            return raster_mode;
        }

        [[gnu::const]]
        auto get_camera_position() const -> math::Vector<f32, 3> {
            return camera_position;
        }

        void rotate_pitch(math::Angle<f32> angle) {
            camera_pitch += angle;
        }

        void rotate_yaw(math::Angle<f32> angle) {
            camera_yaw += angle;
        }

        void rotate_roll(math::Angle<f32> angle) {
            camera_roll += angle;
        }

        [[gnu::const]] // Mark const since this is a pure function and can be optimized away.
        auto rotation_matrix() const -> math::Matrix<f32, 3, 3> {
            using Matrix = math::Matrix<f32, 3, 3>;
            return Matrix::rotation(Matrix::RotationAxis::Pitch, camera_pitch)
                 * Matrix::rotation(Matrix::RotationAxis::Yaw, camera_yaw);
        }

        [[gnu::const]] // Mark const since this is a pure function and can be optimized away.
        auto view_direction() const -> math::Vector<f32, 3> {
            return math::Vector<f32, 3> { 0.f, 0.f, 1.f } * rotation_matrix();
        }

        auto cast_ray(math::Vector<f32, 3> origin, math::Vector<f32, 3> direction) const -> std::optional<Hit> {
            std::optional<Hit> best_hit;

            for (auto const& [shape, material] : object_data) {
                std::visit(
                    [&] (auto const& object) {
                        using T = std::decay_t<decltype(object)>;

                        if constexpr (std::same_as<T, Sphere>) {
                            auto l = origin - object.position;
                            f32 a = direction.dot(direction);
                            f32 b = 2.0f * direction.dot(l);
                            f32 c = l.dot(l) - object.radius * object.radius;

                            f32 disc = b * b - 4 * a * c;
                            if (disc >= 0) {
                                f32 sqrt_disc = std::sqrt(disc);
                                f32 t0 = (-b - sqrt_disc) / (2 * a);
                                f32 t1 = (-b + sqrt_disc) / (2 * a);
                                f32 distance = (t0 > 0) ? t0 : ((t1 > 0) ? t1 : -1);
                                if (distance > 0 and (not best_hit or distance < best_hit->distance)) {
                                    auto hit_point = origin + direction * distance;
                                    best_hit = {
                                        .origin = hit_point,
                                        .normal = (hit_point - object.position).normalized(),
                                        .distance = distance,
                                        .material_index = material
                                    };
                                }
                            }
                        } else if constexpr (std::same_as<T, Plane>) {
                            f32 denom = direction.dot(object.normal);
                            if (std::abs(denom) > 1e-6f) {
                                f32 distance = (object.position - origin).dot(object.normal) / denom;
                                if (distance > 0 and (not best_hit or distance < best_hit->distance)) {
                                    auto hit_point = origin + direction * distance;
                                    best_hit = {
                                        .origin = hit_point,
                                        .normal = object.normal.normalized(),
                                        .distance = distance,
                                        .material_index = material
                                    };
                                }
                            }
                        } else if constexpr (std::same_as<T, Mesh>) {
                            if (auto hit = object.intersect(origin, direction)) {
                                if (not best_hit or hit->distance < best_hit->distance) {
                                    hit->material_index = material;
                                    best_hit = hit;
                                }
                            }
                        }
                    },
                    shape
                );
            }

            return best_hit;
        }

        void draw(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
            const f32 aspect = f32(target.width()) / f32(target.height());
            const f32 half_fov_tan = std::tan(fov.radians() / 2.f);
            const i32 width = target.width();
            const i32 height = target.height();
            const auto rotation_matrix = this->rotation_matrix();

            // This is intended for functional programming of 2d graphics but works to parallelize
            // the raytracer as well, especially nice since std::execution is not a thing in clang.
            // C++ is a sad language, what use is a "standard" if not one implementation is actually complete?
            // Anyhow, this can be made single threaded by using the normal draw::draw adapter, feel free to try.
            // Just remove the _threaded part.
            target | draw::draw_threaded(
                draw::Generator { [&] (i32 x, i32 y) -> draw::Color {
                    if (checkerboard and (x + y + input.counter()) % 2 == 0) return draw::color::CLEAR;

                    const f32 ndc_x = (2.f * (x + .5f) / width - 1.f) * aspect;
                    const f32 ndc_y = (1.f - 2.f * (y + .5f) / height);

                    const f32 px = ndc_x * half_fov_tan;
                    const f32 py = ndc_y * half_fov_tan;

                    math::Vector<f32, 3> forward_ray_dir = { px, py, 1.f };
                    forward_ray_dir = forward_ray_dir.normalized();
                    const auto ray_dir = forward_ray_dir * rotation_matrix;

                    if (const auto hit = cast_ray(camera_position, ray_dir)) {
                        return material_data[hit->material_index]->shade(*hit, *this, 0);
                    } else {
                        return draw::color::CLEAR;
                    }
                } }
                | draw::slice(0, 0, width, height) // Generators are infinite, we only want the area overlapping the target.
            );
        }

// Note that the Vulkan implementation is incomplete and does not do anything yet,
// It's just there for figuring out how to set up compute.
#ifdef ENABLE_VULKAN_SUPPORT
      private:
        vk::UniqueInstance vk_instance;
        vk::UniqueDevice vk_device;
        vk::Queue vk_queue;
        vk::UniqueShaderModule vk_shader_module;

        mutable u32 vk_buffer_width  = 0;
        mutable u32 vk_buffer_height = 0;

        class VulkanUsageError final : std::exception {
            std::string reason;

          public:
            explicit VulkanUsageError(std::string reason) : reason(std::move(reason)) {}

            auto what() const noexcept -> char const* override {
                return reason.c_str();
            };
        };

        void init_vulkan(Io& io) {
            vk::ApplicationInfo app_info {
                "Renderer", VK_MAKE_VERSION(1, 0, 0),
                "Puzel", VK_MAKE_VERSION(1, 0, 0),
                VK_API_VERSION_1_4
            };

            std::array layers = { "VK_LAYER_KHRONOS_validation" };
            std::array extensions = {
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
            };

            vk::InstanceCreateInfo create_info {
                vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
                &app_info,
                layers.size(),
                layers.data(),
                extensions.size(),
                extensions.data()
            };

            vk_instance = vk::createInstanceUnique(create_info);

            std::vector<vk::PhysicalDevice> devices = vk_instance->enumeratePhysicalDevices();
            if (devices.empty()) throw VulkanUsageError("No physical devices found");

            vk::PhysicalDevice physical_device;
            u32 compute_queue_family_index = std::numeric_limits<u32>::max();

            for (auto& dev : devices) {
                auto queueFamilies = dev.getQueueFamilyProperties();
                for (u32 i = 0; i < queueFamilies.size(); i += 1) {
                    if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                        physical_device = dev;
                        compute_queue_family_index = i;
                        goto end_device_loop;
                    }
                }
            }
          end_device_loop:
            if (not physical_device) throw VulkanUsageError("No suitable compute device found");

            f32 priority = 1.f;
            vk::DeviceQueueCreateInfo queue_create_info { {}, compute_queue_family_index, 1, &priority };

            std::array device_extensions = {
                "VK_KHR_portability_subset"
            };

            vk::DeviceCreateInfo device_info { {}, {}, {},
                {}, {},
                device_extensions.size(), device_extensions.data()
            };
            device_info.queueCreateInfoCount = 1;
            device_info.pQueueCreateInfos = &queue_create_info;

            vk_device = physical_device.createDeviceUnique(device_info);
            vk_queue = vk_device->getQueue(compute_queue_family_index, 0);

            auto shader = io.read_file("res/basdf.comp.spv");
            vk::ShaderModuleCreateInfo shader_info { {}, shader.size(), (u32 const*) shader.data() };
            vk_shader_module = vk_device->createShaderModuleUnique(shader_info);

            vk::PhysicalDeviceMemoryProperties memory_properties = physical_device.getMemoryProperties();
        }

        std::optional<std::string> vk_error;
        std::optional<std::string> generic_error;

      public:
        void draw_vulkan(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
            World& self = *const_cast<World*>(this); // Hack to do some interior mutability conveniently.

            if (vk_error or generic_error) {
                auto message = draw::Text(
                    vk_error
                        ? std::format("Vulkan Error: {}", *vk_error)
                        : std::format("Generic Error: {}", *generic_error),
                    font::mine(),
                    draw::color::pico::RED)
                ;
                target
                    | draw::clear(draw::color::BLACK)
                    | draw::draw(message, target.width() - message.width() - 8, target.height() - message.height() - 8);
            } else if (not vk_instance) {
                try {
                    self.init_vulkan(io);
                } catch (vk::Error& e) {
                    self.vk_error = e.what();
                } catch (VulkanUsageError& e) {
                    self.generic_error = e.what();
                } catch (SdlIo::Error& e) {
                    self.generic_error = e.what();
                }
            }

        }
#else
        void draw_vulkan(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
            auto message = draw::Text("Vulkan support not compiled", font::mine(), draw::color::pico::RED);
            target
                | draw::clear(draw::color::BLACK)
                | draw::draw(message, target.width() - message.width() - 8, target.height() - message.height() - 8);
        }
#endif

        void draw_raster(Io& io, rt::Input const& input, draw::Ref<draw::Image> target) const {
            target | draw::clear(background_color);

            auto depth_buffer = std::vector(target.width() * target.height(), std::numeric_limits<f32>::infinity());

            using Matrix = math::Matrix<f32, 4, 4>;
            using Vector = Matrix::Vector;
            const auto transform =
                Matrix::translation(-camera_position.x(), -camera_position.y(), -camera_position.z())
              * Matrix::rotation(Matrix::RotationAxis::Yaw, -camera_yaw)
              * Matrix::rotation(Matrix::RotationAxis::Pitch, -camera_pitch)
              * Matrix::projection(target.width(), target.height(), get_fov(), .1f, 1000.f);

            static constexpr auto sample_texture = [] (draw::SizedPlane auto const& texture, f32 u, f32 v) -> draw::Color {
                return texture.get(u * (texture.width() - 1), v * (texture.height() - 1));
            };

            static constexpr auto edge = [] (Vector a, Vector b, f32 px, f32 py) -> f32 {
                return (px - a.x()) * (b.y() - a.y()) - (py - a.y()) * (b.x() - a.x());
            };

            struct Tri final {
                std::array<Vector, 3> vertices;
                std::array<Vector, 3> world_pos;
                std::array<math::Vector<f32, 3>, 3> uv;
                std::array<math::Vector<f32, 3>, 3> normal;

                math::Vector<f32, 3> tangent;
                math::Vector<f32, 3> bitangent;

                [[gnu::const]]
                constexpr auto inv_w(usize index) -> f32 {
                    return 1.f / vertices.at(index).w();
                }
            };

            const auto width  = target.width();
            const auto height = target.height();

            for (auto const& [shape, material] : object_data) {
                if (const auto mesh = std::get_if<Mesh>(&shape)) {
                    const auto diffuse // Apply fallback pink texture if missing.
                        = [&mesh] -> draw::EitherPlane<draw::Ref<const draw::Image>, draw::FilledRectangle> {
                            if (mesh->diffuse) {
                                return *mesh->diffuse | draw::as_ref();
                            } else {
                                return draw::FilledRectangle {
                                    .w = 10,
                                    .h = 10,
                                    .color = draw::color::pico::PINK
                                };
                            }
                        }();

                    for (auto const& face : mesh->faces) {
                        const auto combined_transform = mesh->local_to_world() * transform;

                        Tri tri = {
                            .vertices = {
                                Vector(mesh->vertices.at(face.at(0).vertex), 1.f) * combined_transform,
                                Vector(mesh->vertices.at(face.at(1).vertex), 1.f) * combined_transform,
                                Vector(mesh->vertices.at(face.at(2).vertex), 1.f) * combined_transform
                            },
                            .world_pos = {
                                Vector(mesh->vertices.at(face.at(0).vertex), 1.f) * mesh->local_to_world(),
                                Vector(mesh->vertices.at(face.at(1).vertex), 1.f) * mesh->local_to_world(),
                                Vector(mesh->vertices.at(face.at(2).vertex), 1.f) * mesh->local_to_world()
                            },
                            .uv = {
                                mesh->texture_coordinates[face.at(0).texture_coordinate],
                                mesh->texture_coordinates[face.at(1).texture_coordinate],
                                mesh->texture_coordinates[face.at(2).texture_coordinate]
                            },
                            .normal = {
                                Vector(mesh->normals[face.at(0).normal], 0.f) * mesh->local_to_world(),
                                Vector(mesh->normals[face.at(1).normal], 0.f) * mesh->local_to_world(),
                                Vector(mesh->normals[face.at(2).normal], 0.f) * mesh->local_to_world()
                            }
                        };

                        const auto p0 = tri.world_pos[0];
                        const auto p1 = tri.world_pos[1];
                        const auto p2 = tri.world_pos[2];

                        const auto uv0 = tri.uv[0];
                        const auto uv1 = tri.uv[1];
                        const auto uv2 = tri.uv[2];

                        const auto e1 = math::Vector<f32, 3>(p1 - p0);
                        const auto e2 = math::Vector<f32, 3>(p2 - p0);

                        const f32 dU1 = uv1[0] - uv0[0];
                        const f32 dV1 = uv1[1] - uv0[1];
                        const f32 dU2 = uv2[0] - uv0[0];
                        const f32 dV2 = uv2[1] - uv0[1];

                        const f32 r = 1.f / (dU1 * dV2 - dU2 * dV1);

                        tri.tangent   = ((e1 * dV2 - e2 * dV1) * r).normalized();
                        tri.bitangent = ((e2 * dU1 - e1 * dU2) * r).normalized();

                        if (tri.vertices[0].w() == 0.f or tri.vertices[1].w() == 0.f or tri.vertices[2].w() == 0.f)
                            continue;

                        { // Remove back faces.
                            const math::Vector<f32, 3> v0 = tri.vertices[0] / tri.vertices[0].w();
                            const math::Vector<f32, 3> v1 = tri.vertices[1] / tri.vertices[1].w();
                            const math::Vector<f32, 3> v2 = tri.vertices[2] / tri.vertices[2].w();

                            const auto e0 = v1 - v0;
                            const auto e1 = v2 - v0;
                            const auto face_normal = e0.cross(e1);

                            // If the face points away then skip.
                            if (face_normal.z() >= 0.f)
                                continue;
                        }

                        // Map triangle to screen-space.
                        for (auto& v : tri.vertices) {
                            v.x() /= v.w(); v.y() /= v.w(); v.z() /= v.w();

                            v.x() = (v.x() * .5f + .5f) * f32(target.width());
                            v.y() = (1.f - (v.y() * .5f + .5f)) * f32(target.height());
                        }

                        if (raster_mode == RasterMode::Wireframe) {
                            target | draw::line(tri.vertices[0].x(), tri.vertices[0].y(), tri.vertices[1].x(), tri.vertices[1].y())
                                   | draw::line(tri.vertices[1].x(), tri.vertices[1].y(), tri.vertices[2].x(), tri.vertices[2].y())
                                   | draw::line(tri.vertices[2].x(), tri.vertices[2].y(), tri.vertices[0].x(), tri.vertices[0].y());
                        } else {
                            // Bounding box
                            const f32 min_x = std::min({ tri.vertices[0].x(), tri.vertices[1].x(), tri.vertices[2].x() });
                            const f32 max_x = std::max({ tri.vertices[0].x(), tri.vertices[1].x(), tri.vertices[2].x() });
                            const f32 min_y = std::min({ tri.vertices[0].y(), tri.vertices[1].y(), tri.vertices[2].y() });
                            const f32 max_y = std::max({ tri.vertices[0].y(), tri.vertices[1].y(), tri.vertices[2].y() });

                            i32 x0 = std::max(0, i32(std::floor(min_x)));
                            i32 x1 = std::min(width - 1, i32(std::ceil(max_x)));
                            i32 y0 = std::max(0, i32(std::floor(min_y)));
                            i32 y1 = std::min(height - 1, i32(std::ceil(max_y)));

                            x0 = std::clamp(x0, 0, width - 1);
                            x1 = std::clamp(x1, 0, width - 1);
                            y0 = std::clamp(y0, 0, height - 1);
                            y1 = std::clamp(y1, 0, height - 1);

                            f32 area = edge(tri.vertices[0], tri.vertices[1], tri.vertices[2].x(), tri.vertices[2].y());
                            if (std::abs(area) < 0.001f) continue;

                            if (tri.vertices[0].z() < .1f) continue;
                            if (tri.vertices[1].z() < .1f) continue;
                            if (tri.vertices[2].z() < .1f) continue;

                            if (tri.vertices[0].w() < 0.f) continue;
                            if (tri.vertices[1].w() < 0.f) continue;
                            if (tri.vertices[2].w() < 0.f) continue;

                            // Iterate box
                            for (i32 y = y0; y <= y1; y += 1) {
                                for (i32 x = x0; x <= x1; x += 1) {
                                    f32 px = x + .5f;
                                    f32 py = y + .5f;

                                    f32 w0 = edge(tri.vertices[1], tri.vertices[2], px, py);
                                    f32 w1 = edge(tri.vertices[2], tri.vertices[0], px, py);
                                    f32 w2 = edge(tri.vertices[0], tri.vertices[1], px, py);

                                    bool inside = (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0);
                                    if (not inside) continue;

                                    // Normalize coordinates
                                    w0 /= area;
                                    w1 /= area;
                                    w2 /= area;

                                    f32 recip_w = w0 * tri.inv_w(0) + w1 * tri.inv_w(1) + w2 * tri.inv_w(2);
                                    if (recip_w <= 0.f) continue;

                                    f32 z_ndc = (w0 * tri.vertices[0].z() * tri.inv_w(0) + w1 * tri.vertices[1].z() * tri.inv_w(1) + w2 * tri.vertices[2].z() * tri.inv_w(2)) / recip_w;

                                    i32 idx = x + y * width;
                                    if (z_ndc >= depth_buffer[idx]) continue;

                                    f32 u = (w0 * tri.uv[0][0] * tri.inv_w(0) + w1 * tri.uv[1][0] * tri.inv_w(1) + w2 * tri.uv[2][0] * tri.inv_w(2)) / recip_w;
                                    f32 v = (w0 * tri.uv[0][1] * tri.inv_w(0) + w1 * tri.uv[1][1] * tri.inv_w(1) + w2 * tri.uv[2][1] * tri.inv_w(2)) / recip_w;

                                    math::Vector<f32, 3> tangent_space_normal = { 0.f, 0.f, 1.f };
                                    math::Vector<f32,3> normal;
                                    if (auto const& normal_map = mesh->normal; normal_map and normal_mapping) {
                                        // Interpolated vertex normal
                                        auto n =
                                            ((tri.normal[0] * (w0 * tri.inv_w(0)) +
                                              tri.normal[1] * (w1 * tri.inv_w(1)) +
                                              tri.normal[2] * (w2 * tri.inv_w(2))) / recip_w).normalized();

                                        auto ncolor = sample_texture(*normal_map, u, 1.f - v);

                                        // Normalize rgb to -1...1
                                        tangent_space_normal = {
                                            ncolor.r / 255.f * 2.f - 1.f,
                                            ncolor.g / 255.f * 2.f - 1.f,
                                            ncolor.b / 255.f * 2.f - 1.f
                                        };

                                        math::Vector<f32, 3> tspace = tangent_space_normal.normalized();

                                        const auto t = tri.tangent;
                                        const auto b = tri.bitangent;

                                        const auto tbn = math::Matrix<f32, 3, 3>({
                                            { t.x(), b.x(), n.x() },
                                            { t.y(), b.y(), n.y() },
                                            { t.z(), b.z(), n.z() }
                                        });

                                        normal = (tspace * tbn).normalized();
                                    } else {
                                        normal =
                                            ((tri.normal[0] * (w0 * tri.inv_w(0)) +
                                              tri.normal[1] * (w1 * tri.inv_w(1)) +
                                              tri.normal[2] * (w2 * tri.inv_w(2))) / recip_w).normalized();
                                    }

                                    math::Vector<f32,3> origin =
                                        (tri.world_pos[0] * (w0 * tri.inv_w(0)) +
                                         tri.world_pos[1] * (w1 * tri.inv_w(1)) +
                                         tri.world_pos[2] * (w2 * tri.inv_w(2))) / recip_w;

                                    auto diffuse_color = sample_texture(diffuse, u, 1.f - v);
                                    math::Vector<f32, 3> out_color;

                                    for (const auto light : lights()) {
                                        auto roughness = .5f;
                                        auto metallic = 0.f;
                                        if (auto const& roughness_map = mesh->gloss) {
                                            roughness = math::normalize(
                                                f32(sample_texture(*roughness_map, u, 1.f - v).r),
                                                0.f, 255.f,
                                                0.f, 1.f
                                            );
                                        }
                                        if (auto const& specular_map = mesh->specular) {
                                            metallic = math::normalize(
                                                f32(sample_texture(*specular_map, u, 1.f - v).r),
                                                0.f, 255.f,
                                                0.f, 1.f
                                            );
                                        }

                                        const auto light_direction = (light.position - origin).normalized();
                                        const auto distance_to_light = (light.position - origin).magnitude();
                                        const auto view_direction = (get_camera_position() - origin).normalized();
                                        const auto half = (view_direction + light_direction).normalized();
                                        const auto base_reflectivity = math::mix(
                                            math::Vector<f32, 3>(.04f),
                                            math::Vector<f32, 3>(Color(diffuse_color)),
                                            metallic
                                        );

                                        const auto normal_distribution
                                            = math::sq(roughness)
                                            / (f32(math::pi) * math::sq(
                                                math::sq(normal.dot(half)) * (math::sq(roughness) - 1.f) + 1.f
                                            ));

                                        // const auto fresnel
                                        //     = base_reflectivity + (math::Vector<f32, 3>(1.f) - base_reflectivity) * std::pow(
                                        //         1.f - std::clamp(half.dot(view_direction), 0.f, 1.f),
                                        //         5.f
                                        //     );

                                        // Fresnel without the powf call.
                                        f32 x = 1.f - std::clamp(half.dot(view_direction), 0.f, 1.f);
                                        f32 x2 = x * x;
                                        f32 x5 = x2 * x2 * x;
                                        const auto fresnel = base_reflectivity + (math::Vector<f32, 3>(1.f) - base_reflectivity) * x5;

                                        const auto direct_k = math::sq(roughness + 1.f) / 8.f;
                                        const auto ndotv = std::clamp(normal.dot(view_direction), 0.f, 1.f);
                                        const auto ndotl = std::clamp(normal.dot(light_direction), 0.f, 1.f);
                                        const auto microfacets
                                            = (ndotv / std::max(.0001f, ndotv * (1.f - direct_k) + direct_k))
                                            * (ndotl / std::max(.0001f, ndotl * (1.f - direct_k) + direct_k));

                                        const auto cook_torrance
                                            = (fresnel * normal_distribution * microfacets)
                                            / (4.f * view_direction.dot(normal) * light_direction.dot(normal));

                                        const auto lambert_diffuse
                                            = math::Vector<f32, 3>(light.color).hadamard(Color(diffuse_color))
                                            * std::max(0.f, normal.dot(light_direction));
                                        const auto diffuse_reflectance = (math::Vector<f32, 3>(1.f) - fresnel) * (1.f - metallic);

                                        out_color += diffuse_reflectance.hadamard(lambert_diffuse)
                                                  + cook_torrance.hadamard(math::Vector<f32, 3>(light.color)) * ndotl;
                                    }

                                    target | draw::pixel(x, y, Color(out_color));
                                    depth_buffer[idx] = z_ndc;
                                }
                            }
                        }
                    }
                }
            }

            if (raster_mode == RasterMode::DepthBuffer) {
                target | draw::draw_threaded(draw::Generator([&] (i32 x, i32 y) -> draw::Color {
                    return draw::Color::gray(255.f - math::normalize(depth_buffer[x + y * width], .1f, .25f, 4.f, 255.f));
                }) | draw::slice(0, 0, width, height));
            }
        }
    };
}
