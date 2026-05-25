#include "instance.hpp"

#define STB_VORBIS_IMPLEMENTATION
#include "stb_vorbis.h"

using namespace rt;

// Okay. So, stb_vorbis commits gettext underscore levels of tragedy.
// It has to go in a new cpp file which slows down compilation, or I can use some other nonsensical workaround.
//
// THE PREPROCESSOR WALL OF SHAME:
// - On Windows SDL defines main even for C++ (because apparently that's superior to just telling MSVC to use main).
// - On Windows Microsoft defines a criminal amount of pure garbage macros.
// - Gettext makes the brilliant decision to define the fucking underscore (yay modern C++ standard collisions).
// - Stb vorbis defines L, R and C and who knows what else. AT LEAST UNDEF THEM AFTERWARDS FOR FUCKS SAKE.
//
// Because I have no clue what other garbage is going to FUCK with templates even if I undef those,
// the safest option is to redeclare what I need myself.
//
// Alternatively SDL does have an audio library, but due to the stupid way SDL uses trampolines for
// everything dead code elimination doesn't work on it, and it's generally used as a massive dynamic library anyway.
// I do not wish to bother with fixing static linkage or adding another massive library just to use one function.
auto SdlIo::perform_read_oggfile(char const* path, u32 frequency) -> std::vector<f32> {
    i32 errval = 0;

    stb_vorbis* vorbis = stb_vorbis_open_filename(path, &errval, nullptr);
    if (not vorbis) throw std::runtime_error("stb error");

    ScopeExit scope_exit_vorbis = [=] { stb_vorbis_close(vorbis); };

    stb_vorbis_info info = stb_vorbis_get_info(vorbis);
    u32 src_channels = info.channels;
    u32 src_freq = info.sample_rate;
    u32 total_samples = stb_vorbis_stream_length_in_samples(vorbis) * src_channels;

    std::vector<f32> raw_decoded(total_samples);

    stb_vorbis_get_samples_float_interleaved(vorbis, src_channels, raw_decoded.data(), total_samples);

    SDL_AudioSpec src_spec {
        .format = SDL_AUDIO_F32,
        .channels = (i32) src_channels,
        .freq = (i32) src_freq
    };

    SDL_AudioSpec dst_spec {
        .format = SDL_AUDIO_F32,
        .channels = 1,
        .freq = (i32) frequency
    };

    if (src_spec.channels == dst_spec.channels && src_spec.freq == dst_spec.freq) return raw_decoded;

    u8* dst_data = nullptr;
    i32 dst_count = 0;

    if (not SDL_ConvertAudioSamples(
        &src_spec, (u8*) raw_decoded.data(), raw_decoded.size() * sizeof(f32),
        &dst_spec, &dst_data, &dst_count
    )) {
        throw Error();
    }

    ScopeExit scope_exit_dst = [=] { SDL_free(dst_data); };

    u32 float_count = dst_count / sizeof(f32);
    f32* float_data = (f32*) dst_data;

    return std::vector<f32>(float_data, float_data + float_count);
}
