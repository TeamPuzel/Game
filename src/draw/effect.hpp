#include "plane.hpp"
#include <math>

// Effects -------------------------------------------------------------------------------------------------------------
namespace draw {
    namespace cfg {
        struct CrtConfig {
            f32 phosphor = 1.0f;
            f32 vscanlines = 0.0f;

            f32 input_gamma  = 2.5f;
            f32 output_gamma = 2.2f;

            f32 sharpness = 1.0f;

            f32 color_boost = 1.5f;
            f32 red_boost   = 1.0f;
            f32 green_boost = 1.0f;
            f32 blue_boost  = 1.0f;

            f32 scanlines_strength = 0.5f;

            f32 beam_min_width = 0.86f;
            f32 beam_max_width = 1.0f;

            f32 anti_ringing = 0.8f;
        };
    }

    template <SizedPlane T> struct CrtEffect final {
        T inner;
        i32 scale;
        cfg::CrtConfig config;

        constexpr auto width() const noexcept(noexcept(inner.width())) -> i32 {
            return inner.width() * scale;
        }

        constexpr auto height() const noexcept(noexcept(inner.height())) -> i32 {
            return inner.height() * scale;
        }

      private:
        static inline auto to_float(Color c) noexcept -> std::array<f32,3> {
            return { c.r / 255.f, c.g / 255.f, c.b / 255.f };
        }

        static inline auto from_float(f32 r,f32 g,f32 b) noexcept -> Color {
            auto clamp = [](f32 v)->u8 {
                return (u8)std::clamp(v*255.f,0.f,255.f);
            };
            return Color::rgba(clamp(r),clamp(g),clamp(b));
        }

        static inline auto gamma_in(std::array<f32,3>& c,f32 g) noexcept {
            c[0]=pow(c[0],g);
            c[1]=pow(c[1],g);
            c[2]=pow(c[2],g);
        }

        static inline auto gamma_out(std::array<f32,3>& c,f32 g) noexcept {
            f32 inv = 1.f/g;
            c[0]=pow(c[0],inv);
            c[1]=pow(c[1],inv);
            c[2]=pow(c[2],inv);
        }

      public:
        inline auto get(i32 x, i32 y) const noexcept(noexcept(inner.get(x, y))) -> Color {
            const f32 fx = (f32)x / scale;
            const f32 fy = (f32)y / scale;

            const f32 fractx = fx - floor(fx);
            const f32 fracty = fy - floor(fy);

            const i32 ix = std::clamp((i32)floor(fx),0,inner.width()-1);
            const i32 iy = std::clamp((i32)floor(fy),0,inner.height()-1);

            auto sample=[&](i32 ox,i32 oy){
                ox = std::clamp(ix+ox,0,inner.width()-1);
                oy = std::clamp(iy+oy,0,inner.height()-1);

                auto c=to_float(inner.get(ox,oy));
                gamma_in(c,config.input_gamma);
                return c;
            };

            // Neighbor taps (equivalent layout to shader)
            auto c00=sample(-1,-1);
            auto c01=sample(0,-1);
            auto c02=sample(1,-1);
            auto c03=sample(2,-1);

            auto c10=sample(-1,0);
            auto c11=sample(0,0);
            auto c12=sample(1,0);
            auto c13=sample(2,0);

            std::array<f32,3> min_sample;
            std::array<f32,3> max_sample;

            for(int i=0;i<3;i++){
                min_sample[i]=std::min({c01[i],c11[i],c02[i],c12[i]});
                max_sample[i]=std::max({c01[i],c11[i],c02[i],c12[i]});
            }

            // Cubic interpolation weights (Catmull-Rom)
            auto cubic=[&](f32 A,f32 B,f32 C,f32 D,f32 t){
                f32 a = -0.5f*A + 1.5f*B - 1.5f*C + 0.5f*D;
                f32 b = A - 2.5f*B + 2.f*C - 0.5f*D;
                f32 c = -0.5f*A + 0.5f*C;
                f32 d = B;
                return ((a*t + b)*t + c)*t + d;
            };

            std::array<f32,3> color0;
            std::array<f32,3> color1;

            for(int i=0;i<3;i++){
                color0[i]=cubic(c00[i],c01[i],c02[i],c03[i],fractx);
                color1[i]=cubic(c10[i],c11[i],c12[i],c13[i],fractx);
            }

            // Anti-ringing clamp
            for(int i=0;i<3;i++){
                auto aux=color0[i];
                color0[i]=std::clamp(color0[i],min_sample[i],max_sample[i]);
                color0[i]=aux + config.anti_ringing*(color0[i]-aux);

                aux=color1[i];
                color1[i]=std::clamp(color1[i],min_sample[i],max_sample[i]);
                color1[i]=aux + config.anti_ringing*(color1[i]-aux);
            }

            f32 pos0=fracty;
            f32 pos1=1-fracty;

            std::array<f32,3> lum0,lum1;

            for(int i=0;i<3;i++){
                lum0[i]=config.beam_min_width +
                    (config.beam_max_width-config.beam_min_width)*color0[i];

                lum1[i]=config.beam_min_width +
                    (config.beam_max_width-config.beam_min_width)*color1[i];
            }

            std::array<f32,3> d0,d1;

            for(int i=0;i<3;i++){

                d0[i]=std::clamp(pos0/(lum0[i]+1e-7f),0.f,1.f);
                d1[i]=std::clamp(pos1/(lum1[i]+1e-7f),0.f,1.f);

                d0[i]=exp(-10.f*config.scanlines_strength*d0[i]*d0[i]);
                d1[i]=exp(-10.f*config.scanlines_strength*d1[i]*d1[i]);
            }

            std::array<f32,3> color;

            for(int i=0;i<3;i++)
                color[i]=std::clamp(color0[i]*d0[i]+color1[i]*d1[i],0.f,1.f);

            color[0]*=config.color_boost*config.red_boost;
            color[1]*=config.color_boost*config.green_boost;
            color[2]*=config.color_boost*config.blue_boost;

            // Dot mask
            if(config.phosphor>0.f){
                f32 mod = ((x+y)&1);
                std::array<f32,3> mask =
                    mod ? std::array<f32,3>{0.7f,1.f,0.7f}
                        : std::array<f32,3>{1.f,0.7f,1.f};

                for(int i=0;i<3;i++)
                    color[i]*=(1-config.phosphor)+config.phosphor*mask[i];
            }

            gamma_out(color,config.output_gamma);

            return from_float(color[0],color[1],color[2]);
        }
    };

    namespace adapt {
        struct CrtEffect final {
            i32 scale;
            cfg::CrtConfig cfg;

            template <SizedPlane T> constexpr auto operator()(T inner) const noexcept -> draw::CrtEffect<T> {
                return draw::CrtEffect<T> { inner, scale, cfg };
            }
        };
    }

    constexpr adapt::CrtEffect crt_effect(i32 scale, cfg::CrtConfig cfg = {}) noexcept {
        return adapt::CrtEffect { scale, cfg };
    }
}
