//
// Created by Ninter6 on 2024/11/23.
//

#include "serena.hpp"

using mathpls::vec2;

constexpr float dt = 0.02f;
constexpr vec2 g = {0, -10};

struct Point {
    vec2 pos = 0;
    vec2 acc = g;
    vec2 dx = g*dt*dt/2;
};

constexpr int num_points = 100;
Point points[num_points]{};

void init() {
    auto line = 10;
    vec2 s = {-9};
    for (int i = 0; i < line; ++i)
        for (int j = 0; j < num_points / line; ++j)
            points[i * line + j].pos = {s.x + (float)j * 2.f, s.y + (float)i * 2.f};
}

void step() {
    for (auto&& i : points) i.acc = g;

    // TODO

    for (auto&& i : points) {
        i.dx += i.acc * dt*dt;
        i.pos += i.dx;

        if (i.pos.y < -15) {
            i.pos.y = -15;
            i.dx.y = -i.dx.y * .707f;
        }

        if (i.pos.x < -24) i.pos.x = -18;
        if (i.pos.x > 24) i.pos.x = 18;
    }
}

int main() {
    st::init_window("cloth", {1280, 800});

    st::drawer dr{1024, 256};
    dr.set_camera({ .half_extent = {24, 15} });
    dr.tile_draw_circle(dr.create("circle"), {1});

    init();

    while (!st::window_should_close()) {
        step();

        for (auto&& i : points)
            dr.draw_tile("circle", { .pos = i.pos });

        dr.submit();
        st::next_frame({});
    }
}