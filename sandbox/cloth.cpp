//
// Created by Ninter6 on 2024/11/23.
//

#include "serena.hpp"

using namespace mathpls;

constexpr float fps = 100;
constexpr float spf = 5;
constexpr vec2 g = {0, -10};

constexpr float K = 50;

struct Point {
    vec2 pos = 0;
    vec2 vel = 0;
    vec2 acc = g;

    bool fixed = false;
};

constexpr int num_points = 100;
Point points[num_points]{};

struct Link {
    Link(int a, int b, float R) : a(a), b(b), R(R) {}
    int a, b;
    float R;
};
std::vector<Link> link;

void init() {
    auto line = 10;
    vec2 s = {-9};
    for (int i = 0; i < line; ++i)
        for (int j = 0; j < num_points / line; ++j)
            points[i * line + j].pos = {s.x + (float)j * 2.f, s.y + (float)i * 2.f};


    for (int i = 0; i < line - 1; ++i)
        for (int j = 0; j < num_points / line - 1; ++j) {
            auto lb = i * line + j;
            auto rb = i * line + j + 1;
            auto lt = (i + 1) * line + j;
            auto rt = (i + 1) * line + j + 1;
            link.emplace_back(lb, rb, 1.9f);
            link.emplace_back(lb, lt, 1.9f);
            link.emplace_back(rb, rt, 1.9f);
            link.emplace_back(lt, rt, 1.9f);
            link.emplace_back(lb, rt, 2.8f);
            link.emplace_back(rb, lt, 2.8f);
        }

    points[num_points - 1].fixed = true;
    points[num_points - line].fixed = true;
}

void calcu_force() {
    for (auto&& [a, b, R] : link) {
        auto& p1 = points[a];
        auto& p2 = points[b];

        auto f = K * (R - distance(p1.pos, p2.pos));

        auto dir = p1.pos - p2.pos;
        p1.acc += f * dir;
        p2.acc -= f * dir;
    }
}

void step(float dt) {
    for (auto&& i : points) {
        i.acc = g;
        i.vel *= .999f;
    }

    calcu_force();

    for (auto&& i : points) {
        if (i.fixed) continue;
        i.vel += i.acc * dt * .5f;
        i.pos += i.vel * dt + i.acc * dt*dt * .5f;
    }

    calcu_force();

    for (auto&& i : points) {
        if (i.fixed) continue;
        i.vel += i.acc * dt * .5f;
    }

    for (auto&& i : points) {
        if (i.fixed) continue;
        if (i.pos.y < -15) {
            i.pos.y = -15;
            i.vel.y = -i.vel.y * .707f;
        }

        if (i.pos.x < -24) i.pos.x = -24;
        if (i.pos.x > 24) i.pos.x = 24;
    }
}

int main() {
    st::init_window("cloth", {1280, 800}, true);

    st::drawer dr{1024, 256};
    dr.set_camera({ .half_extent = {24, 15} });
    dr.tile_draw_circle(dr.create("circle"), {1});

    init();

    while (!st::window_should_close()) {
        for (int i = 0; i < (int)spf; i++)
            step(1.f / fps / spf);

        for (auto&& i : points)
            dr.draw_tile("circle", { .pos = i.pos, .scale = .5f });
        for (auto&& [a, b, _] : link)
            dr.draw_line(points[a].pos, points[b].pos, .1f);

        dr.submit();
        st::next_frame({});
    }
}