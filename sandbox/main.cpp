#include <iostream>
#include <vector>

#include "serena.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    st::init_window("serena test", {800, 500});

    st::drawer dr{512, 256};
    auto rect = dr.create();
    dr.tile_draw_rectangle(rect, 1.618, {1, 1, 0, 1});
    for (int i = 0; i < 5000; i++) {
        mathpls::vec2 p = mathpls::random::rand_vec2() * mathpls::random::rand11<float>() * 30;
        dr.draw(rect, {.pos = p, .rotate = mathpls::random::rand11<float>()*3.14f}, {.sampler = st::sampler_type::linear});
    }
    auto circle = dr.create();
    dr.tile_clear(circle);
    dr.tile_draw_circle(circle, {0, 0, 1, 1});
    dr.draw(circle, {.scale = 2.f}, {.sampler = st::sampler_type::linear});

    dr.set_camera({0, {40, 25}, M_PI_4});

    while (!st::window_should_close()) {
        auto err=glGetError();
        assert(err == GL_NO_ERROR);
        dr.submit_without_clear();
        st::next_frame({1});
    }

    return 0;
}
