#include <iostream>

#include "serena.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    st::init_window("serena test", {800, 500});

    st::drawer dr{1024, 256};

//    stbi_set_flip_vertically_on_load(true);
//    int w, h, channel;
//    auto data = stbi_load(ROOT"heart.png", &w, &h, &channel, 0);
//    assert(data);
//
//    auto img = dr.create("img");
//    dr.tile_image(img, {w, h}, channel, data);
//    auto cp = dr.create("cp");
//    dr.tile_draw_circle(cp, {.8f, .2f, 1, 1});
//    dr.tile_draw_tile(cp, img, {.scale = .5, .rotate = M_PI_4});
//    dr.draw_tile(cp, {.scale = 2});
//
//    dr.create(st::auto_delete, "feel free");
//    dr.tile_draw_rectangle("feel free", 1, {1, 0, 0, 1});
//    mathpls::vec2 p[] = {{-3}, {-2, -3}, {-3.5, 0}};
//    dr.draw_polygon("feel free", p, {.color = {.1f, 1,.7f, 1}});

    dr.set_camera({{0}, {4, 2.5}, 0});

    while (!st::window_should_close()) {
        auto err=glGetError();
        assert(err == GL_NO_ERROR);

        mathpls::vec2 p[] = {{0, 1}, {-1}, {1, -1}};
        dr.draw_polygon(p, {.color = {.1f, 1,.7f, 1}});

        if (st::is_mouse_button_down(0)) {
            dr.draw_line(dr.get_drag_start_position(0, st::world_space), dr.get_mouse_pos(st::world_space),.05, {1, 0, 0, 1});
        }

        if (st::is_mouse_button_long_pressed(1)) {
            dr.draw_line({-1}, {1}, .02, {0, 1, 1, 1});
            dr.draw_line({1, -1}, {-1, 1}, .02, {0, 1, 1, 1});
        }

        dr.submit();
        st::next_frame({1});
    }

    return 0;
}
