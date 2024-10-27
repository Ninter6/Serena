//
// Created by Ninter6 on 2024/10/21.
//

#ifndef SERENA_HPP
#define SERENA_HPP

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include "mathpls.h"

#include <span>
#include <string>
#include <memory>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <algorithm>

namespace st {

#define STR(x) #x
#define XSTR(x) STR(x)

#define SERENA_OPENGL_VERSION_MAJOR 4
#define SERENA_OPENGL_VERSION_MINOR 1
#define SERENA_GLSL_VERSION \
    "#version " XSTR(SERENA_OPENGL_VERSION_MAJOR) XSTR(SERENA_OPENGL_VERSION_MINOR) "0 core"

using extent = mathpls::ivec2;

using tile = uint32_t;

struct serena_global_context {
    GLFWwindow* window{};
    extent window_extent{};

    GLint max_texture_size{};

    // sampler
    GLuint nearest_sampler{};
    GLuint linear_sampler{};

    // mesh
    GLuint rect_vbo{};

    // shaders
    GLuint simple_sh{};
    GLuint polygon_sh{};
    GLuint tile_circle_sh{};
    GLuint tile_rect_sh{};
};
inline serena_global_context* global_context = nullptr;

namespace detail {

inline void init_glfw_window(const char* title, int width, int height, bool msaa_enable = false) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, SERENA_OPENGL_VERSION_MAJOR);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, SERENA_OPENGL_VERSION_MINOR);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (msaa_enable) glfwWindowHint(GLFW_SAMPLES, 4);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    global_context->window = glfwCreateWindow(width/2, height/2, title, nullptr, nullptr);
#else
    global_context->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
#endif

    if (!global_context->window) throw std::runtime_error("failed to create GLFW window");
    glfwMakeContextCurrent(global_context->window);
}

inline void init_glad() {
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("failed to initialize GLAD");
}

inline auto compile_shader(const char* vsh, const char* fsh) {
    const auto v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vsh, nullptr);
    glCompileShader(v);

    const auto f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fsh, nullptr);
    glCompileShader(f);

    const auto p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);

    GLint success;
    glGetProgramiv(p, GL_LINK_STATUS, &success);
    if(!success) {
        GLint length;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &length);
        std::string infoLog("\0", length);
        glGetProgramInfoLog(p, 512, nullptr, infoLog.data());
        throw std::runtime_error("Failed to link shader:\n" + infoLog);
    }

    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

constexpr auto simple_vsh = SERENA_GLSL_VERSION R"(
layout (location = 0) in vec2 vert;
layout (location = 1) in vec2 uv;
layout (location = 2) in uint tile;
layout (location = 3) in vec4 color;
layout (location = 4) in vec2 pos;
layout (location = 5) in vec2 scale;
layout (location = 6) in float rotate;
layout (location = 7) in float spl;
out vec2 frag_uv;
flat out vec4 frag_color;
flat out float frag_spl;
layout (std140) uniform UBO { // default binding to 0
    vec2 tile_count;
    vec2 cam_centre;
    vec2 cam_half_extent;
    float cam_rotate;
};
mat3 view = mat3( cos(cam_rotate)/cam_half_extent.x,-sin(cam_rotate)/cam_half_extent.y, 0,
                  sin(cam_rotate)/cam_half_extent.x, cos(cam_rotate)/cam_half_extent.y, 0,
                 -dot(vec2(cos(cam_rotate), sin(cam_rotate)), cam_centre)/cam_half_extent.x, dot(vec2(sin(cam_rotate),-cos(cam_rotate)), cam_centre)/cam_half_extent.y, 0 );
void main() {
    float t = float(tile) / tile_count.x;
    frag_uv = uv / tile_count + vec2(fract(t), floor(t) / tile_count.y);
    frag_color = color;
    frag_spl = spl;
    mat2 M = mat2( cos(rotate), sin(rotate),
                  -sin(rotate), cos(rotate) );
    vec2 p = M * ((vert - 0.5) * scale) + pos;
    gl_Position = vec4(view * vec3(p, 1), 1);
})";

constexpr auto simple_fsh = SERENA_GLSL_VERSION R"(
in vec2 frag_uv;
flat in vec4 frag_color;
flat in float frag_spl;
out vec4 color;
uniform sampler2D nboard; // default binding to 0
uniform sampler2D lboard;
void main() {
    color = mix(texture(nboard, frag_uv), texture(lboard, frag_uv), frag_spl) * frag_color;
})";

constexpr auto polygon_vsh = SERENA_GLSL_VERSION R"(
layout (location = 0) in vec2 pos;
uniform mat3 M;
void main() {
    gl_Position = vec4(M * vec3(pos, 1), 1);
})";

constexpr auto polygon_fsh = SERENA_GLSL_VERSION R"(
uniform vec4 color;
out vec4 out_color;
void main () {
    out_color = color;
})";

constexpr auto rect_vsh = SERENA_GLSL_VERSION R"(
layout (location = 0) in vec2 vert;
layout (location = 1) in vec2 uv;
out vec2 frag_uv;
void main() {
    frag_uv = vert * 2 - 1;
    gl_Position = vec4(frag_uv, 0, 1);
})";

constexpr auto tile_circle_fsh = SERENA_GLSL_VERSION R"(
in vec2 frag_uv;
out vec4 out_color;
uniform vec4 color;
void main() {
    if (length(frag_uv) > 1) discard;
    out_color = color;
})";

constexpr auto tile_rect_fsh = SERENA_GLSL_VERSION R"(
in vec2 frag_uv;
out vec4 out_color;
uniform vec4 color;
uniform float w_h;
void main() {
    if (frag_uv.x*frag_uv.x > w_h*w_h || frag_uv.y*frag_uv.y*w_h*w_h > 1) discard;
    out_color = color;
})";

inline void init_shader() {
    global_context->simple_sh = compile_shader(simple_vsh, simple_fsh);
    glUseProgram(global_context->simple_sh);
    glUniform1i(glGetUniformLocation(global_context->simple_sh, "lboard"), 1);

    global_context->polygon_sh = compile_shader(polygon_vsh, polygon_fsh);
    global_context->tile_circle_sh = compile_shader(rect_vsh, tile_circle_fsh);
    global_context->tile_rect_sh = compile_shader(rect_vsh, tile_rect_fsh);
}

inline void init_sampler() {
    GLuint temp[2]{};
    glGenSamplers(2, temp);
    global_context->nearest_sampler = temp[0];
    global_context->linear_sampler = temp[1];

    glSamplerParameteri(temp[0], GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(temp[0], GL_TEXTURE_WRAP_T, GL_REPEAT);
    glSamplerParameteri(temp[0], GL_TEXTURE_WRAP_R, GL_REPEAT);
    glSamplerParameteri(temp[0], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(temp[0], GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glSamplerParameteri(temp[1], GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(temp[1], GL_TEXTURE_WRAP_T, GL_REPEAT);
    glSamplerParameteri(temp[1], GL_TEXTURE_WRAP_R, GL_REPEAT);
    glSamplerParameteri(temp[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(temp[1], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

struct unit {
    tile id{};
    mathpls::vec4 color{};
    float sampler{};
    mathpls::vec2 pos{};
    mathpls::vec2 scale{1};
    float rotate{};
};

inline void bind_unit_attr(GLuint vbo) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribIPointer(2, 1, GL_UNSIGNED_INT, sizeof(unit), nullptr);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 4, GL_FLOAT, false, sizeof(unit), (void*)offsetof(unit, color));
    glVertexAttribDivisor(3, 1);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 2, GL_FLOAT, false, sizeof(unit), (void*)offsetof(unit, pos));
    glVertexAttribDivisor(4, 1);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(5, 2, GL_FLOAT, false, sizeof(unit), (void*)offsetof(unit, scale));
    glVertexAttribDivisor(5, 1);
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(6, 1, GL_FLOAT, false, sizeof(unit), (void*)offsetof(unit, rotate));
    glVertexAttribDivisor(6, 1);
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(7, 1, GL_FLOAT, false, sizeof(unit), (void*)offsetof(unit, sampler));
    glVertexAttribDivisor(7, 1);
    glEnableVertexAttribArray(7);
}

inline std::pair<GLuint, GLuint> create_polygon_mesh(const void* data, size_t size) {
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)size, data, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, false, 2*sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    return {vao, vbo};
}

constexpr size_t min_unit_size = 256;
class unit_pool {
    void init(size_t n) {
        buf_size = n*sizeof(unit);
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)buf_size, nullptr, GL_DYNAMIC_DRAW);
        mapped = (unit*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        bind_attr();
    }

    void delete_buffer() {
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glDeleteBuffers(1, &buffer);
        mapped = nullptr;
    }

    void bind_attr() const {
        glBindVertexArray(vao);
        detail::bind_unit_attr(buffer);
    }

public:
    explicit unit_pool(GLuint vao, size_t units = min_unit_size) : vao(vao) {
        init(std::max(min_unit_size, units));
    }

    unit_pool(unit_pool&&) = delete;

    ~unit_pool() { delete_buffer(); }

    [[nodiscard]] size_t size() const { return unit_size; }
    [[nodiscard]] size_t capacity() const { return buf_size / sizeof(unit); }
    [[nodiscard]] bool empty() const { return size() == 0; }

    void reserve(size_t n) {
        if (n <= capacity()) return;
        GLuint new_buf;
        size_t new_size = n*sizeof(unit);
        glGenBuffers(1, &new_buf);
        glBindBuffer(GL_ARRAY_BUFFER, new_buf);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)new_size, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_COPY_READ_BUFFER, buffer);
        glBindBuffer(GL_COPY_WRITE_BUFFER, new_buf);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, (GLsizeiptr)(unit_size*sizeof(unit)));
        glDeleteBuffers(1, &buffer);
        buf_size = new_size;
        buffer = new_buf;
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        mapped = (unit*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
        bind_attr();
    }

    void resize(size_t n) {
        if (n <= size()) return;
        reserve(n);
        unit_size = n;
    }

    [[nodiscard]] unit& at(size_t n) const {
        assert(n < size());
        return mapped[n];
    }

    template <class...Args>
    void emplace_back(Args&&...args) {
        assert(size() <= capacity());
        if (size() == capacity())
            reserve(size()<<1);
        mapped[unit_size++] = {std::forward<decltype(args)>(args)...};
    }

    void pop_back(size_t n = 1) {
        assert(size() >= n);
        unit_size -= n;
    }

    void clear() { unit_size = 0; }

    void erase(size_t n) {
        if (n >= size()) return;
        std::rotate(mapped+n, mapped+n+1, mapped+unit_size);
        pop_back();
    }

    void erase(size_t begin, size_t end) {
        assert(begin <= size() && end <= size() && begin <= end);
        std::rotate(mapped+begin, mapped+end, mapped+unit_size);
        pop_back(end-begin);
    }

private:
    GLuint buffer{};
    unit* mapped{};
    size_t buf_size{};
    size_t unit_size{};

    GLuint vao;
};

struct drawer_ubo {
    // board
    mathpls::vec2 tile_count{};
    // camera
    mathpls::vec2 centre{};
    mathpls::vec2 half_extent{};
    float rotate{};
};

}

inline void init_window(std::string_view title, extent window_size, bool msaa_enable = false) {
    assert(!global_context && "excess initialization");
    global_context = new serena_global_context;
    global_context->window_extent = window_size;

    detail::init_glfw_window(title.data(), window_size.x, window_size.y, msaa_enable);
    detail::init_glad();

    glViewport(0, 0, window_size.x, window_size.y);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &global_context->max_texture_size);
    std::clog << "Max Texture Size: " << global_context->max_texture_size << '\n';

    constexpr float rect[] = { 0,0, 1,1, 1,0, 0,1, 1,1, 0,0 };
    glGenBuffers(1, &global_context->rect_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, global_context->rect_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rect), rect, GL_STATIC_DRAW);

    detail::init_shader();
}

inline bool window_should_close() {
    return glfwWindowShouldClose(global_context->window);
}

inline void window_close() {
    delete global_context;
    global_context = nullptr;
    glfwTerminate();
}

inline void next_frame() {
    glfwPollEvents();
    glfwSwapBuffers(global_context->window);
}

inline void next_frame(mathpls::vec4 clear_color) {
    next_frame();
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
    glClear(GL_COLOR_BUFFER_BIT);
}

struct transform {
    mathpls::vec2 pos{};
    mathpls::vec2 scale{1};
    float rotate{};
};

enum class sampler_type {
    nearest = 0, linear = 1
};
struct material {
    mathpls::vec4 color{1};
    sampler_type sampler = sampler_type::nearest;
};

struct camera {
    mathpls::vec2 centre{};
    mathpls::vec2 half_extent{};
    float rotate{};
};

class drawer {
    void init() {
        init_board();
        init_fbo();
        init_vao();
        init_ubo();
        units = std::make_unique<detail::unit_pool>(vao);
    }

    void init_board() {
        glGenTextures(1, &board_tex);
        glBindTexture(GL_TEXTURE_2D, board_tex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, board_size.x, board_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    void init_fbo() {
        glGenFramebuffers(1, &board_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, board_tex, 0);

        constexpr GLenum draw_buffers[] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, draw_buffers);

        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void init_vao() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, global_context->rect_vbo);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
    }

    void init_ubo() {
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(detail::drawer_ubo), nullptr, GL_DYNAMIC_DRAW);
        ubo_mapped = (detail::drawer_ubo*)glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
        ubo_mapped->tile_count = board_size / tile_size;
    }

    [[nodiscard]] mathpls::ivec2 tile_offset(tile id) const {
        const auto [cx, cy] = (board_size / tile_size).asArray;
        auto i = (id % cx), j = (id / cx);
        assert(j < cy);
        return {(int)i * tile_size.x, (int)j * tile_size.y};
    }

    static void screen_viewport() {
        const auto [w, h] = global_context->window_extent.asArray;
        glViewport(0, 0, w, h);
    }
    void tile_viewport(tile id) const {
        const auto [x, y] = tile_offset(id).asArray;
        glViewport(x, y, tile_size.x, tile_size.y);
    }

public:
    explicit drawer(extent board_size = 8192, extent tile_size = 256)
        : board_size(board_size), tile_size(tile_size) {
        assert(global_context && "init window first");
        assert(board_size.x % tile_size.x == 0 && board_size.y % tile_size.y == 0);
        init();
    }

    drawer(drawer&&) = delete;

    ~drawer() {
        glDeleteFramebuffers(1, &board_fbo);
        glDeleteTextures(1, &board_tex);
        glDeleteVertexArrays(1, &vao);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glUnmapBuffer(GL_UNIFORM_BUFFER);
        glDeleteBuffers(1, &ubo);
    }

    tile create() {
        return curr_tile++;
    }

    void set_camera(const camera& cam) {
        ubo_mapped->centre = cam.centre;
        ubo_mapped->half_extent = cam.half_extent;
        ubo_mapped->rotate = cam.rotate;
    }

    void draw(tile id, const transform& t, const material& material = {}) const {
        auto& [pos, scale, rotate] = t;
        auto& [color, sampler] = material;
        units->emplace_back(id, color, (float)sampler, pos, scale, rotate);
    }

    void submit() const {
        submit_without_clear();
        units->clear();
    }

    void submit_without_clear() const {
        if (units->empty()) return;
        screen_viewport();
        glEnable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, board_tex);
        glBindSampler(0, global_context->nearest_sampler);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, board_tex);
        glBindSampler(1, global_context->linear_sampler);
        glBindVertexArray(vao);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
        glUseProgram(global_context->simple_sh);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, (GLsizei)units->size());
        glBindSampler(0, 0);
        glBindSampler(1, 0);
        glDisable(GL_BLEND);
    }

    void tile_clear(tile id, const mathpls::vec4 clear_color = {}) {
        tile_draw_rectangle(id, 1, clear_color);
    }

    void tile_draw_polygon(tile id, std::span<const mathpls::vec2> points, const mathpls::vec4& color) {
        assert(points.size() > 2);
        auto [pvao, pvbo] = detail::create_polygon_mesh(points.data(), points.size()*sizeof(mathpls::vec2));

        mathpls::vec2 mn{}, mx{};
        for (auto&& i : points) {
            mn.x = std::min(i.x, mn.x);
            mn.y = std::min(i.y, mn.y);
            mx.x = std::max(i.x, mx.x);
            mx.y = std::max(i.y, mx.y);
        }
        auto C = (mn + mx) * .5f;
        auto E = mx - C;
        auto S = std::max(E.x, E.y);
        mathpls::mat3 M = {
            mathpls::vec3{1/S, 0, 0},
            mathpls::vec3{0, 1/S, 0},
            mathpls::vec3{-C.x/S, -C.y/S, 0}
        };

        tile_viewport(id);
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glUseProgram(global_context->polygon_sh);
        glUniformMatrix3fv(glGetUniformLocation(global_context->polygon_sh, "M"), 1, false, M.value_ptr());
        glUniform4fv(glGetUniformLocation(global_context->polygon_sh, "color"),1 , color.value_ptr());
        glBindVertexArray(pvao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)points.size());
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteBuffers(1, &pvbo);
        glDeleteVertexArrays(1, &pvao);
    }

    void tile_draw_circle(tile id, const mathpls::vec4& color) const {
        tile_viewport(id);
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glUseProgram(global_context->tile_circle_sh);
        glUniform4fv(glGetUniformLocation(global_context->tile_circle_sh, "color"), 1, color.value_ptr());
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void tile_draw_rectangle(tile id, float w_h, const mathpls::vec4& color) {
        tile_viewport(id);
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glUseProgram(global_context->tile_rect_sh);
        glUniform4fv(glGetUniformLocation(global_context->tile_rect_sh, "color"), 1, color.value_ptr());
        glUniform1f(glGetUniformLocation(global_context->tile_rect_sh, "w_h"), w_h);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

private:
    GLuint board_tex{};
    extent board_size{};
    extent tile_size{};

    GLuint board_fbo{};

    GLuint ubo{};
    detail::drawer_ubo* ubo_mapped{};

    std::unique_ptr<detail::unit_pool> units{};
    GLuint vao{};

    tile curr_tile = 0;
};

}

#undef STR
#undef XSTR

#endif //SERENA_HPP
