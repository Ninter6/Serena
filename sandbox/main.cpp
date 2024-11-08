#include <iostream>

#include "serena.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    st::init_window("serena test", {800, 500});

    st::drawer dr{1024, 256};

    stbi_set_flip_vertically_on_load(true);
    int w, h, channel;
    auto data = stbi_load(ROOT"heart.png", &w, &h, &channel, 0);
    assert(data);

    auto img = dr.create("img");
    dr.tile_image(img, {w, h}, channel, data);
    auto cp = dr.create("cp");
    dr.tile_draw_circle(cp, {.8f, .2f, 1, 1});
    dr.tile_draw_tile(cp, img, {.scale = .5, .rotate = M_PI_4});
    dr.draw_tile(cp, {.scale = 2});

    dr.create("feel free");
    mathpls::vec2 p[] = {{-3}, {-2, -3}, {-3.5, -1}};
    dr.draw_polygon("feel free", p, {.color = {.1f, 1,.7f, 1}});

    dr.set_camera({{0}, {4, 2.5}, 0});

    while (!st::window_should_close()) {
        auto err=glGetError();
        assert(err == GL_NO_ERROR);
        dr.submit_without_clear();
        st::next_frame({1});
    }

    return 0;
}
