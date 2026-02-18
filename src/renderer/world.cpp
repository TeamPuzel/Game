#include "world.hpp"

using namespace renderer;

auto LambertMaterial::shade(Hit hit, World const& world, u32 depth) const -> renderer::Color {
    using Vector = math::Vector<f32, 3>;

    Vector out_color;

    for (const auto light : world.lights()) {
        const auto light_direction = (light.position - hit.origin).normalized();
        const auto distance_to_light = (light.position - hit.origin).magnitude();
        const auto view_direction = (world.get_camera_position() - hit.origin).normalized();
        const auto half = (view_direction + light_direction).normalized();

        if (world.get_shadows()) [[likely]] {
            const auto shadow_origin = hit.origin + hit.normal.normalized() * 0.001f;
            if (auto shadow_hit = world.cast_ray(shadow_origin, light_direction))
                if (shadow_hit->distance < distance_to_light) continue;
        }

        const auto lambert_diffuse
            = Vector(light.color).hadamard(color)
            * std::max(0.f, hit.normal.dot(light_direction));

        out_color += lambert_diffuse * diffuse_reflectance;
    }

    return out_color;
}

auto BsdfMaterial::shade(Hit hit, World const& world, u32 depth) const -> renderer::Color {
    using Vector = math::Vector<f32, 3>;
    constexpr static f32 EPSILON = .001f;

    const Vector base_color = color;
    const auto roughness = this->roughness * this->roughness;

    Vector out_color;

    const auto base_reflectivity = math::mix(Vector(.04f), base_color, metallic);
    const auto view_direction = (world.get_camera_position() - hit.origin).normalized();

    // Specular and diffuse pass ---------------------------------------------------------------------------------------
    for (const auto light : world.lights()) {
        const auto light_direction = (light.position - hit.origin).normalized();
        const auto distance_to_light = (light.position - hit.origin).magnitude();
        const auto half = (view_direction + light_direction).normalized();

        // if (world.get_shadows()) [[likely]] {
        //     const auto shadow_origin = hit.origin + hit.normal.normalized() * 0.001f;
        //     if (auto shadow_hit = world.cast_ray(shadow_origin, light_direction))
        //         if (shadow_hit->distance < distance_to_light) continue;
        // }

        const auto normal_distribution
            = math::sq(roughness)
            / (f32(math::pi) * math::sq(
                math::sq(hit.normal.dot(half)) * (math::sq(roughness) - 1.f) + 1.f
            ));

        const auto fresnel
            = base_reflectivity + (Vector(1.f) - base_reflectivity) * std::pow(
                1.f - std::clamp(half.dot(view_direction), 0.f, 1.f),
                5.f
            );

        const auto direct_k = math::sq(roughness + 1.f) / 8.f;
        const auto ndotv = std::clamp(hit.normal.dot(view_direction), 0.f, 1.f);
        const auto ndotl = std::clamp(hit.normal.dot(light_direction), 0.f, 1.f);
        const auto microfacets
            = (ndotv / std::max(EPSILON, ndotv * (1.f - direct_k) + direct_k))
            * (ndotl / std::max(EPSILON, ndotl * (1.f - direct_k) + direct_k));

        const auto cook_torrance
            = (fresnel * normal_distribution * microfacets)
            / (4.f * view_direction.dot(hit.normal) * light_direction.dot(hit.normal));

        const auto lambert_diffuse
            = Vector(light.color).hadamard(base_color)
            * std::max(0.f, hit.normal.dot(light_direction));
        const auto diffuse_reflectance = (Vector(1.f) - fresnel) * (1.f - metallic);

        switch (world.get_bsdf_mode()) {
            [[likely]]
            case BsdfMaterial::Mode::Default:
                out_color += diffuse_reflectance.hadamard(lambert_diffuse)
                          +  cook_torrance.hadamard(Vector(light.color)) * ndotl;
                break;
            case BsdfMaterial::Mode::Diffuse:
                out_color += lambert_diffuse;
                break;
            case BsdfMaterial::Mode::CookTorrance:
                out_color += cook_torrance;
                break;
            case BsdfMaterial::Mode::Fresnel:
                out_color += fresnel;
                break;
            case BsdfMaterial::Mode::NormalDistribution:
                out_color += normal_distribution;
                break;
            case BsdfMaterial::Mode::Microfacets:
                out_color += microfacets;
                break;
        }
    }

    // Reflection pass -------------------------------------------------------------------------------------------------
    if (false and depth < 4 and metallic > 0.f and (1.f - roughness) > EPSILON) {
        const auto reflect_direction
            = (-view_direction + hit.normal * (2.f * view_direction.dot(hit.normal))).normalized();
        const auto reflect_origin = hit.origin + hit.normal * EPSILON;

        if (auto next_hit = world.cast_ray(reflect_origin, reflect_direction)) {
            auto reflected_color = world.material(next_hit->material_index).shade(*next_hit, world, depth + 1);

            const auto reflection_strength = (1.f - roughness);

            const auto specular = Vector(reflected_color).hadamard(
                base_reflectivity + (Vector(1.f) - base_reflectivity) * std::pow(
                    1.f - std::clamp(hit.normal.dot(view_direction), 0.f, 1.f),
                    5.f
                )
            ).hadamard(math::mix(Vector(1.f), base_color, metallic));

            out_color += specular * (metallic * reflection_strength);
        } else {
            const auto specular = Vector(world.get_background_color()).hadamard(
                base_reflectivity + (Vector(1.f) - base_reflectivity) * std::pow(
                    1.f - std::clamp(hit.normal.dot((world.get_camera_position() - hit.origin).normalized()), 0.f, 1.f),
                    5.f
                )
            ).hadamard(math::mix(Vector(1.f), base_color, metallic));

            const auto reflection_strength = (1.f - roughness);

            out_color += specular * (metallic * reflection_strength);
        }
    }

    // Global illumination pass ----------------------------------------------------------------------------------------

    constexpr static auto build_tangent_space = [] (const Vector& n, Vector& t, Vector& b) {
        if (std::fabs(n.x()) > std::fabs(n.z())) {
            t = Vector(-n.y(), n.x(), 0.f).normalized();
        } else {
            t = Vector(0.f, -n.z(), n.y()).normalized();
        }
        b = n.cross(t);
    };

    constexpr static auto sample_cosine_hemisphere_diffuse_only = [] (f32 u1, f32 u2) -> Vector {
        const f32 r = std::sqrt(u1);
        const f32 phi = 2.f * math::pi * u2;
        const f32 x = r * std::cos(phi);
        const f32 z = r * std::sin(phi);
        const f32 y = std::sqrt(std::max(0.f, 1.f - u1));
        return { x, y, z };
    };

    const static auto sample_cosine_hemisphere = [] (f32 u1, f32 u2, f32 roughness) -> Vector {
        const f32 r = std::sqrt(u1) * roughness; // Scale radius by roughness.
        const f32 phi = 2.f * math::pi * u2;

        const f32 x = r * std::cos(phi);
        const f32 z = r * std::sin(phi);
        const f32 y = std::sqrt(std::max(0.f, 1.f - x * x - z * z)); // Maintain hemisphere.
        return { x, y, z };
    };

    Vector gi_color;

    if (world.get_gi_mode() == GiMode::Simple) {
        constexpr static i32 GI_RING_COUNT = 32;
        constexpr static i32 GI_SAMPLES_PER_RING = 32;
        constexpr static i32 GI_SAMPLE_COUNT = GI_RING_COUNT * GI_SAMPLES_PER_RING;
        constexpr static f32 GI_CLAMP = 1.f;
        constexpr static i32 GI_MAX_DEPTH = 1;
        constexpr static auto clamp_fn = [] (f32 e) { return std::min(GI_CLAMP, e); };

        if (depth < GI_MAX_DEPTH) {
            const auto gi_origin = hit.origin + hit.normal * EPSILON;

            std::array<Vector, GI_SAMPLE_COUNT> gi_directions = [roughness] {
                std::array<Vector, GI_SAMPLE_COUNT> directions;
                i32 idx = 0;

                for (i32 r = 0; r < GI_RING_COUNT; r += 1) {
                    const f32 t = (r + .5f) / GI_RING_COUNT;
                    const f32 theta = t * (math::pi * .5f);

                    for (i32 s = 0; s < GI_SAMPLES_PER_RING; s += 1) {
                        const f32 phi = (2.f * math::pi) * (f32(s) / GI_SAMPLES_PER_RING);

                        const f32 sin_theta = std::sin(theta);
                        const f32 cos_theta = std::cos(theta);

                        // instead of the old uniform cosine hemisphere, use roughness-biased:
                        const f32 u1 = theta / (math::pi * 0.5f); // map theta to [0,1]
                        const f32 u2 = s / f32(GI_SAMPLES_PER_RING);

                        directions[idx++] = sample_cosine_hemisphere(u1, u2, roughness);
                    }
                }
                return directions;
            }();

            Vector tangent, bitangent;
            build_tangent_space(hit.normal, tangent, bitangent);

            for (auto const& dir : gi_directions) {
                const auto world_dir = tangent * dir.x() + hit.normal * dir.y() + bitangent * dir.z();

                if (auto bounce_hit = world.cast_ray(gi_origin, world_dir)) {
                    const auto bounce_color = world.material(bounce_hit->material_index).shade(*bounce_hit, world, depth + 1);

                    const f32 cosine = std::max(0.f, world_dir.dot(hit.normal));
                    gi_color += (base_color.hadamard(bounce_color) * cosine).map(clamp_fn);
                } else {
                    gi_color += base_color.hadamard(world.get_background_color());
                }
            }

            // Average samples.
            gi_color /= GI_SAMPLE_COUNT;
        }
    }

    return out_color + gi_color + emissive;
}
