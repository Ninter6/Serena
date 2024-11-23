//
// Created by Ninter6 on 2024/10/21.
//

#ifndef SERENA_HPP
#define SERENA_HPP

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include "mathpls.h"

#include <set>
#include <span>
#include <string>
#include <memory>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <unordered_map>

namespace st {

#define STR(x) #x
#define XSTR(x) STR(x)

#define SERENA_OPENGL_VERSION_MAJOR 4
#define SERENA_OPENGL_VERSION_MINOR 1
#define SERENA_GLSL_VERSION \
    "#version " XSTR(SERENA_OPENGL_VERSION_MAJOR) XSTR(SERENA_OPENGL_VERSION_MINOR) "0 core"

using extent = mathpls::ivec2;

using tile = uint32_t;

constexpr tile null_tile = UINT32_MAX;

class mouse_manager {
private:
    static constexpr int mouse_button_last = 3;
    // 0: left mouse
    // 1: right mouse
    // 2: middle mouse

    bool button_states[mouse_button_last]{};
    bool button_pressed[mouse_button_last]{};
    bool button_released[mouse_button_last]{};
    double last_press_time[mouse_button_last]{};
    mathpls::dvec2 drag_start_positions[mouse_button_last]{};
    mathpls::dvec2 current_position{};
    double current_time{};

public:
    mouse_manager() = default;

    void update(GLFWwindow* window) {
        current_time = glfwGetTime();
        // 更新鼠标按键状态
        for (int i = 0; i < mouse_button_last; ++i) {
            if (glfwGetMouseButton(window, i) == GLFW_PRESS) {
                if (!button_states[i]) {
                    button_states[i] = true;
                    button_pressed[i] = true;
                    last_press_time[i] = current_time;
                    drag_start_positions[i] = current_position;
                } else {
                    button_pressed[i] = false;
                }
            } else {
                if (button_states[i]) {
                    button_states[i] = false;
                    button_released[i] = true;
                } else {
                    button_released[i] = false;
                }
            }
        }

        mathpls::ivec2 window_size;
        glfwGetCursorPos(window, &current_position.x, &current_position.y);
        glfwGetWindowSize(window, &window_size.x, &window_size.y);
        current_position /= (mathpls::dvec2)window_size;
    }

    bool is_button_pressed(int button) {
        return button_pressed[button];
    }

    bool is_button_released(int button) {
        return button_released[button];
    }

    bool is_button_down(int button) {
        return button_states[button];
    }

    bool is_button_long_pressed(int button, double long_press_duration = 0.5) {
        return button_states[button] && (current_time - last_press_time[button] >= long_press_duration);
    }

    mathpls::dvec2 get_position() {
        return current_position;
    }

    mathpls::dvec2 get_drag_start_position(int button) {
        return drag_start_positions[button];
    }

    void reset_button_state(int button) {
        button_states[button] = false;
        button_pressed[button] = false;
        button_released[button] = false;
    }
};

struct serena_global_context {
    GLFWwindow* window{};
    extent window_extent{};

    mouse_manager mouse{};

    GLint max_texture_size{};

    // sampler
    GLuint nearest_sampler{};
    GLuint linear_sampler{};

    // mesh
    GLuint rect_vbo{};

    // shaders
    GLuint simple_sh{};
    GLuint stupid_sh{};
    GLuint polygon_sh{};
    GLuint tile_circle_sh{};
    GLuint tile_rect_sh{};
};
inline serena_global_context* global_context = nullptr;

namespace detail {

template <class First, class Second>
struct basic_compressed_pair : First, Second {
    basic_compressed_pair() = default;
    basic_compressed_pair(First&& f, Second&& s) : First(std::forward<First>(f)), Second(std::forward<Second>(s)) {}

    [[nodiscard]] First& first() { return *this; }
    [[nodiscard]] const First& first() const { return *this; }
    [[nodiscard]] Second& second() { return *this; }
    [[nodiscard]] const Second& second() const { return *this; }
};

template <class First, class Second>
struct compressed_pair : private basic_compressed_pair<First, Second> {
    using basic_compressed_pair<First, Second>::basic_compressed_pair;
    using basic_compressed_pair<First, Second>::first;
    using basic_compressed_pair<First, Second>::second;
};

template <class T, class Deref, class Res = std::invoke_result_t<Deref, typename std::vector<T>::const_iterator>>
class dense_set_it_warp {
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::remove_reference_t<Res>;
    using pointer = std::remove_reference_t<Res>*;
    using reference = std::remove_reference_t<Res>&;

    explicit dense_set_it_warp(typename std::vector<T>::const_iterator it) : it_(it) {}

    Res operator*() const { static Deref d{}; return d(it_); }

    dense_set_it_warp& operator++() { return ++it_, *this; }

    dense_set_it_warp operator++(int) {
        dense_set_it_warp tmp = *this;
        return ++(*this), tmp;
    }

    dense_set_it_warp& operator--() { return --it_, *this; }

    dense_set_it_warp operator--(int) {
        dense_set_it_warp tmp = *this;
        return --(*this), tmp;
    }

    dense_set_it_warp& operator+=(difference_type n) {
        it_ += n;
        return *this;
    }

    dense_set_it_warp operator+(difference_type n) const {
        return dense_set_it_warp(it_ + n);
    }

    dense_set_it_warp& operator-=(difference_type n) {
        it_ -= n;
        return *this;
    }

    dense_set_it_warp operator-(difference_type n) const {
        return dense_set_it_warp(it_ - n);
    }

    difference_type operator-(const dense_set_it_warp& other) const {
        return it_ - other.it_;
    }

    bool operator==(const dense_set_it_warp& other) const {
        return it_ == other.it_;
    }

    bool operator!=(const dense_set_it_warp& other) const {
        return it_ != other.it_;
    }

    bool operator<(const dense_set_it_warp& other) const {
        return it_ < other.it_;
    }

    bool operator<=(const dense_set_it_warp& other) const {
        return it_ <= other.it_;
    }

    bool operator>(const dense_set_it_warp& other) const {
        return it_ > other.it_;
    }

    bool operator>=(const dense_set_it_warp& other) const {
        return it_ >= other.it_;
    }

private:
    typename std::vector<T>::const_iterator it_;
};

template <class T, class Hash = std::hash<T>, class Eq = std::equal_to<T>, class Alloc = std::allocator<T>>
class dense_set {
    static constexpr size_t null_index = SIZE_T_MAX;
    static constexpr float hash_k = .707f;

    struct node_t {
        template <class...Args>
        explicit node_t(Args&&...args) : value{std::forward<Args>(args)...} {}
        T value;
        size_t next = null_index;
    };

    using alloc_traits = std::allocator_traits<Alloc>;
    using sparse_container = std::vector<size_t, typename alloc_traits::template rebind_alloc<size_t>>;
    using packed_container = std::vector<node_t, typename alloc_traits::template rebind_alloc<node_t>>;

    struct node_deref { const T& operator()(const typename packed_container::const_iterator& it) const {
        return it->value;
    }};

//    [[nodiscard]] auto index_to_iterator(size_t index)  { return begin() + index; }
    [[nodiscard]] auto index_to_iterator(size_t index) const { return cbegin() + index; }

    [[nodiscard]] size_t hash_of(size_t i) const { return sparse_.second()(packed_.first()[i].value); }
    void set_hash(size_t hash, size_t index) {
        assert(sparse_.first().size() > 0);
        hash %= sparse_.first().size();
        sparse_.first()[hash] = index;
    }

    template <class...Args>
    void emplace_back(Args&&...args) {
        packed_.first().emplace_back(std::forward<Args>(args)...);
        length_++;
    }
    template <class...Args>
    void replace_back(Args&&...args) {
        packed_.first()[length_++] = node_t{std::forward<Args>(args)...};
    }

    [[nodiscard]] size_t hash_list_back(size_t hash) const {
        hash %= sparse_.first().size();
        size_t back = sparse_.first()[hash];
        if (back != null_index)
            while (packed_.first()[back].next != null_index)
                back = packed_.first()[back].next;
        return back;
    }
    [[nodiscard]] size_t hash_list_pre(size_t index) const {
        auto hash = this->hash_of(index) % sparse_.first().size();
        size_t pre = sparse_.first()[hash];
        if (pre != index)
            while (packed_.first()[pre].next != index)
                assert(pre != null_index), pre = packed_.first()[pre].next;
        return pre;
    }

    void rehash() {
        sparse_.first().assign(static_cast<size_t>(length_ * 2 / hash_k), null_index);
        for (size_t i = 0; i < length_; i++)
            sparse_emplace(i);
    }

    // need: false, needn't: true
    bool rehash_if_need() {
        if (length_ < sparse_.first().size() * hash_k)
            return true;
        rehash();
        return false;
    }

    void sparse_emplace(size_t id) {
        auto hs = hash_of(id);
        if (auto back = hash_list_back(hs); back == null_index) {
            set_hash(hs, id);
            packed_.first()[id].next = null_index;
        } else {
            packed_.first()[back].next = id;
        }
    }

    void isolate(size_t index) {
        auto pre = hash_list_pre(index);
        assert(pre != null_index);
        if (pre == index)
            set_hash(hash_of(index), packed_.first()[index].next);
        else
            packed_.first()[pre].next = packed_.first()[index].next;
        packed_.first()[index].next = null_index;
    }

    void swap_only(size_t index) {
        isolate(index);
        if (index != --length_) {
            auto pre = hash_list_pre(length_);
            assert(pre != null_index);
            if (pre == length_) set_hash(hash_of(length_), index);
            else packed_.first()[pre].next = index;
            std::swap(packed_.first()[index], packed_.first()[length_]);
        }
    }

    [[nodiscard]] size_t find_index(const T& key) const {
        if (sparse_.first().empty()) return null_index;
        auto hs = sparse_.second()(key) % sparse_.first().size();
        auto i = sparse_.first()[hs];
        while (i != null_index && !packed_.second()(packed_.first()[i].value, key))
            i = packed_.first()[i].next;
        return i;
    }

public:
    using iterator = dense_set_it_warp<node_t, node_deref>;
    using const_iterator = iterator;

    dense_set() = default;

    [[nodiscard]] size_t size() const { return length_; }
    [[nodiscard]] size_t free_size() const { return packed_.first().size() - length_; }
    [[nodiscard]] bool empty() const { return length_ == 0; }

    [[nodiscard]] size_t capacity() const { return packed_.first().capacity(); }
    void reserve(size_t n) { packed_.first().reserve(n); }

    template <class...Args>
    iterator emplace(Args&&...args) {
        size_t index = length_;
        if (free_size() == 0) {
            emplace_back(std::forward<Args>(args)...);
        } else {
            replace_back(std::forward<Args>(args)...);
        }

        if (rehash_if_need())
            sparse_emplace(index);

        return index_to_iterator(index);
    }

    iterator push(const T& v) { return emplace(v); }

    void erase(iterator it) {
        if (it == end()) return;
        auto index = std::distance(begin(), it);
        assert(index < length_);
        swap_only(index);
    }

    void erase(iterator first, iterator last) {
        assert(first <= last);
        while (last != first) erase(--last);
    }

    void erase(const T& key) {
        if (auto it = find(key); it != end())
            erase(it);
    }

    void pop_back() { erase(index_to_iterator(length_ - 1)); }

    void clear() {
        std::uninitialized_fill(sparse_.first().begin(), sparse_.first().end(), null_index);
        length_ = 0;
    }

    void free_clear() {
        packed_.first().erase(packed_.first().begin() + length_, packed_.first().end());
    }

    void shrink_to_fit() {
        free_clear();
        packed_.first().shrink_to_fit();
    }

    const T& undo_pop() {
        assert(free_size() > 0);
        sparse_emplace(length_++);
        return back();
    }

    [[nodiscard]] iterator find(const T& key) {
        auto i = find_index(key);
        return i == null_index ? end() : index_to_iterator(i);
    }
    [[nodiscard]] const_iterator find(const T& key) const {
        auto i = find_index(key);
        return i == null_index ? end() : index_to_iterator(i);
    }

    [[nodiscard]] bool contains(const T& key) const { return find(key) != end(); }

    [[nodiscard]] const T& front() const { return packed_.first().front().value; }
    [[nodiscard]] const T& back() const { return packed_.first()[length_ - 1].value; }

    // [[nodiscard]] iterator begin() { return iterator{packed_.first().cbegin()}; }
    // [[nodiscard]] iterator end() { return begin() + length_; }
    [[nodiscard]] const_iterator begin() const { return const_iterator{packed_.first().cbegin()}; }
    [[nodiscard]] const_iterator end() const { return begin() + length_; }
    // [[nodiscard]] auto rbegin() { return std::make_reverse_iterator(end()); }
    // [[nodiscard]] auto rend() { return std::make_reverse_iterator(begin()); }
    [[nodiscard]] auto rbegin() const { return std::make_reverse_iterator(end()); }
    [[nodiscard]] auto rend() const { return std::make_reverse_iterator(begin()); }
    [[nodiscard]] const_iterator cbegin() const { return begin(); }
    [[nodiscard]] const_iterator cend() const { return end(); }
    [[nodiscard]] auto crbegin() const { return rbegin(); }
    [[nodiscard]] auto crend() const { return rend(); }

private:
    compressed_pair<sparse_container, Hash> sparse_;
    compressed_pair<packed_container, Eq>   packed_;
    size_t length_{};
};

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

constexpr auto stupid_vsh = SERENA_GLSL_VERSION R"(
layout (location = 0) in vec2 vert;
layout (location = 1) in vec2 uv;
uniform uint tile;
uniform vec4 color;
uniform vec2 pos;
uniform vec2 scale;
uniform float rotate;
uniform float spl;
out vec2 frag_uv;
flat out vec4 frag_color;
flat out float frag_spl;
layout (std140) uniform UBO { // default binding to 0
    vec2 tile_count;
    vec2 cam_centre;
    vec2 cam_half_extent;
    float cam_rotate;
};
void main() {
    float t = float(tile) / tile_count.x;
    frag_uv = uv / tile_count + vec2(fract(t), floor(t) / tile_count.y);
    frag_color = color;
    frag_spl = spl;
    mat2 M = mat2( cos(rotate), sin(rotate),
                  -sin(rotate), cos(rotate) );
    vec2 p = M * ((vert * 2 - 1) * scale) + pos;
    gl_Position = vec4(p, 0, 1);
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
    global_context->stupid_sh = compile_shader(stupid_vsh, simple_fsh);
    glUseProgram(global_context->stupid_sh);
    glUniform1i(glGetUniformLocation(global_context->stupid_sh, "lboard"), 1);

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
//    std::clog << "Max Texture Size: " << global_context->max_texture_size << '\n';

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
    global_context->mouse.update(global_context->window);
}

inline void next_frame(mathpls::vec4 clear_color) {
    next_frame();
    glClearColor(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
    glClear(GL_COLOR_BUFFER_BIT);
}

inline bool is_mouse_button_down(int button) {
    return global_context->mouse.is_button_down(button);
}

inline bool is_mouse_button_pressed(int button) {
    return global_context->mouse.is_button_pressed(button);
}

inline bool is_mouse_button_released(int button) {
    return global_context->mouse.is_button_released(button);
}

inline bool is_mouse_button_long_pressed(int button, double long_press_duration = 0.5) {
    return global_context->mouse.is_button_long_pressed(button, long_press_duration);
}

inline mathpls::dvec2 get_mouse_pos() {
    return global_context->mouse.get_position();
}

inline mathpls::dvec2 get_mouse_drag_start_pos(int button) {
    return global_context->mouse.get_drag_start_position(button);
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

class draw_board {
    void init() {
        init_board();
        init_fbo();
        init_vao();
        init_ubo();
        units = std::make_unique<detail::unit_pool>(vao);
        clear_all_tiles();
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
        ubo_mapped = (detail::drawer_ubo*)glMapBuffer(GL_UNIFORM_BUFFER, GL_READ_WRITE);
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
    explicit draw_board(extent board_size = 8192, extent tile_size = 256)
        : board_size(board_size), tile_size(tile_size) {
        assert(global_context && "init window first");
        assert(board_size.x % tile_size.x == 0 && board_size.y % tile_size.y == 0);
        init();
    }

    draw_board(draw_board&&) = delete;

    ~draw_board() {
        glDeleteFramebuffers(1, &board_fbo);
        glDeleteTextures(1, &board_tex);
        glDeleteVertexArrays(1, &vao);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glUnmapBuffer(GL_UNIFORM_BUFFER);
        glDeleteBuffers(1, &ubo);
    }

    tile create() {
        auto [x, y] = (board_size / tile_size).asArray;
        return curr_tile < x*y ? curr_tile++ : null_tile;
    }

    void set_camera(const camera& cam) {
        ubo_mapped->centre = cam.centre;
        ubo_mapped->half_extent = cam.half_extent;
        ubo_mapped->rotate = cam.rotate;
    }

    [[nodiscard]] camera get_camera() const {
        return {ubo_mapped->centre, ubo_mapped->half_extent, ubo_mapped->rotate};
    }

    void draw(tile id, const transform& t, const material& m = {}) const {
        auto& [pos, scale, rotate] = t;
        auto& [color, sampler] = m;
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
        auto [x, y] = tile_offset(id).asArray;
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glEnable(GL_SCISSOR_TEST);
        glScissor(x, y, tile_size.x, tile_size.y);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void tile_image(tile id, extent image_size, int channel, uint8_t* data) {
        GLenum format = GL_RGBA;
        auto [x, y] = tile_offset(id).asArray;
        switch (channel) {
            case 1: format = GL_RED; break;
            case 2: format = GL_RG; break;
            case 3: format = GL_RGB; break;
            case 4: format = GL_RGBA; break;
            default: assert(false);
        }
        if (image_size == tile_size) {
            glBindTexture(GL_TEXTURE_2D, board_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, image_size.x, image_size.y, format, GL_UNSIGNED_BYTE, data);
        } else {
            GLuint buf_tex;
            glGenTextures(1, &buf_tex);
            glBindTexture(GL_TEXTURE_2D, buf_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, image_size.x, image_size.y, 0, format, GL_UNSIGNED_BYTE, data);

            GLuint buf_fbo;
            glGenFramebuffers(1, &buf_fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, buf_fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf_tex, 0);
            assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, buf_fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, board_fbo);
            glBlitFramebuffer(0, 0, image_size.x, image_size.y,
                              x, y,x+tile_size.x,y+tile_size.y,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR);

            glDeleteFramebuffers(1, &buf_fbo);
            glDeleteTextures(1, &buf_tex);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }

    void tile_draw_tile(tile dst, tile src) {
        assert(dst != src);
        auto [dx, dy] = tile_offset(dst).asArray;
        auto [sx, sy] = tile_offset(src).asArray;
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glBindTexture(GL_TEXTURE_2D, board_tex);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, dx, dy, sx, sy, tile_size.x, tile_size.y);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void tile_draw_tile(tile dst, tile src, const transform& t, const material& m = {}) {
        tile_viewport(dst);
        glEnable(GL_BLEND);
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, board_tex);
        glBindSampler(0, global_context->nearest_sampler);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, board_tex);
        glBindSampler(1, global_context->linear_sampler);
        glBindVertexArray(vao);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
        glUseProgram(global_context->stupid_sh);
        glUniform1ui(glGetUniformLocation(global_context->stupid_sh, "tile"), src);
        glUniform2fv(glGetUniformLocation(global_context->stupid_sh, "pos"), 1, t.pos.value_ptr());
        glUniform2fv(glGetUniformLocation(global_context->stupid_sh, "scale"), 1, t.scale.value_ptr());
        glUniform1f(glGetUniformLocation(global_context->stupid_sh, "rotate"), t.rotate);
        glUniform1f(glGetUniformLocation(global_context->stupid_sh, "spl"), (float)m.sampler);
        glUniform4fv(glGetUniformLocation(global_context->stupid_sh, "color"), 1, m.color.value_ptr());
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindSampler(0, 0);
        glBindSampler(1, 0);
        glDisable(GL_BLEND);
    }

    transform tile_draw_polygon(tile id, std::span<const mathpls::vec2> points, const mathpls::vec4& color) {
        assert(points.size() > 2);
        auto [pvao, pvbo] = detail::create_polygon_mesh(points.data(), points.size()*sizeof(mathpls::vec2));

        mathpls::vec2 mn{3.402823E38}, mx{-3.40282E38};
        for (auto&& i : points) {
            mn.x = std::min(i.x, mn.x);
            mn.y = std::min(i.y, mn.y);
            mx.x = std::max(i.x, mx.x);
            mx.y = std::max(i.y, mx.y);
        }
        auto C = (mn + mx) * .5f;
        auto E = mx - C;
        mathpls::mat3 M = {
            mathpls::vec3{1/E.x, 0, 0},
            mathpls::vec3{0, 1/E.y, 0},
            mathpls::vec3{-C.x/E.x, -C.y/E.y, 0}
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

        return { .pos = C, .scale = E*2 };
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

    void clear_all_tiles() const {
        glBindFramebuffer(GL_FRAMEBUFFER, board_fbo);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
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

struct auto_delete_flag_t {};
constexpr auto_delete_flag_t auto_delete;

enum mouse_mode {
    world_space, screen_space
};

class drawer {
public:
    drawer(extent board_size, extent tile_size) {
        board = std::make_unique<draw_board>(board_size, tile_size);
        auto [x, y] = (board_size / tile_size).asArray;
        max_tile_size = x*y;
    }

    [[nodiscard]] mathpls::dvec2 get_mouse_pos(mouse_mode mode) const {
        auto c = st::get_mouse_pos();
        switch (mode) {
            case world_space:
                c = (c * 2 - 1) * (mathpls::dvec2)board->get_camera().half_extent;
                c.y *= -1;
                return c;
            case screen_space: return c; // from (0, 0) to (1, 1)
        }
    }

    [[nodiscard]] mathpls::dvec2 get_drag_start_position(int button, mouse_mode mode) const {
        auto c = get_mouse_drag_start_pos(button);
        switch (mode) {
            case world_space:
                c = (c * 2 - 1) * (mathpls::dvec2)board->get_camera().half_extent;
                c.y *= -1;
                return c;
            case screen_space: return c; // from (0, 0) to (1, 1)
        }
    }

    tile create(const std::string& name) {
        names.push(name);
        if (ids.free_size() > 0)
            return ids.undo_pop();
        else {
            auto id = board->create();
            assert(id != null_tile);
            ids.push(id);
            return id;
        }
    }
    tile create(auto_delete_flag_t, const std::string& name = "__AUtO_dElEtE") {
        auto id = create(name);
        auto_delete_tiles.push_back(id);
        return id;
    }

    void destroy(const std::string& name) {
        destroy(get_id(name));
    }
    void destroy(tile id) {
        assert(ids.contains(id));
        board->tile_clear(id);
        const auto index = std::distance(ids.begin(), ids.find(id));
        ids.erase(ids.begin() + index);
        names.erase(names.begin() + index);
    }

    [[nodiscard]] std::string get_name(tile id) const {
        assert(ids.contains(id));
        const auto index = std::distance(ids.begin(), ids.find(id));
        return *(names.begin() + index);
    }
    [[nodiscard]] tile get_id(const std::string& name) const {
        assert(names.contains(name));
        const auto index = std::distance(names.begin(), names.find(name));
        return *(ids.begin() + index);
    }

    void rename(tile id, const std::string& new_name) {
        const auto index = std::distance(ids.begin(), ids.find(id));
        ids.erase(ids.begin() + index);
        names.erase(names.begin() + index);
        create(new_name);
    }
    void rename(const std::string& name, const std::string& new_name) {
        const auto index = std::distance(names.begin(), names.find(name));
        ids.erase(ids.begin() + index);
        names.erase(names.begin() + index);
        create(new_name);
    }

    [[nodiscard]] bool exist(const std::string& name) const {
        return names.contains(name);
    }

    [[nodiscard]] size_t size() const {
        return ids.size();
    }
    [[nodiscard]] size_t remain() const {
        return max_tile_size - ids.size();
    }

    void clear() {
        ids.clear();
        names.clear();
        board->clear_all_tiles();
        auto_delete_tiles.clear();
    }

    void draw_line(tile buf_tile, mathpls::vec2 a, mathpls::vec2 b, float thickness, mathpls::vec4 color, bool clear_tile = true) {
        if (clear_tile) board->tile_clear(buf_tile, {1});
        auto [dx, dy] = (b - a).asArray;
        board->draw(buf_tile, {
            .pos = (a + b) * .5f,
            .scale = { mathpls::distance(a, b), thickness },
            .rotate = std::atan2(dy, dx)
        }, { .color = color });
    }
    void draw_line(const std::string& buf_tile_name, mathpls::vec2 a, mathpls::vec2 b, float thickness, mathpls::vec4 color, bool clear_tile = true) {
        draw_line(get_id(buf_tile_name), a, b, thickness, color, clear_tile);
    }
    void draw_line(mathpls::vec2 a, mathpls::vec2 b, float thickness, mathpls::vec4 color) {
        if (exist("__AUtO_dElEtE_whItE_blOck"))
            draw_line("__AUtO_dElEtE_whItE_blOck", a, b, thickness, color, false);
        else
            draw_line(create(auto_delete, "__AUtO_dElEtE_whItE_blOck"), a, b, thickness, color);
    }

    void draw_polygon(tile buf_tile, std::span<const mathpls::vec2> points, const material& m) {
        const auto& t = board->tile_draw_polygon(buf_tile, points, m.color);
        draw_tile(buf_tile, t, {.sampler = m.sampler});
    }
    void draw_polygon(const std::string& buf_tile_name, std::span<const mathpls::vec2> points, const material& m) {
        draw_polygon(get_id(buf_tile_name), points, m);
    }
    void draw_polygon(std::span<const mathpls::vec2> points, const material& m) {
        draw_polygon(create(auto_delete), points, m);
    }

    void submit() {
        board->submit();
        for (auto id : auto_delete_tiles)
            destroy(id);
        auto_delete_tiles.clear();
    }
    void submit_without_clear() const { board->submit_without_clear(); }

    void set_camera(const camera& cam) { board->set_camera(cam); }

    void draw_tile(tile id, const transform& t, const material& m = {}) { board->draw(id, t, m); }
    void draw_tile(const std::string& name, const transform& t, const material& m = {}) { board->draw(get_id(name), t, m); }

    void tile_clear(tile id) { board->tile_clear(id); }
    void tile_clear(const std::string& name) { board->tile_clear(get_id(name)); }

    void tile_image(tile id, extent image_size, int channel, uint8_t* data) { board->tile_image(id, image_size, channel, data); }
    void tile_image(const std::string& name, extent image_size, int channel, uint8_t* data) { board->tile_image(get_id(name), image_size, channel, data); }

    void tile_draw_tile(tile dst, tile src) { board->tile_draw_tile(dst, src); }
    void tile_draw_tile(const std::string& dst, const std::string& src) { board->tile_draw_tile(get_id(dst), get_id(src)); }
    void tile_draw_tile(tile dst, tile src, const transform& t, const material& m = {}) { board->tile_draw_tile(dst, src, t, m); }
    void tile_draw_tile(const std::string& dst, const std::string& src, const transform& t, const material& m = {}) { board->tile_draw_tile(get_id(dst), get_id(src), t, m); }

    void tile_draw_polygon(tile id, std::span<const mathpls::vec2> points, const mathpls::vec4& color) { board->tile_draw_polygon(id, points, color); }
    void tile_draw_polygon(const std::string& name, std::span<const mathpls::vec2> points, const mathpls::vec4& color) { board->tile_draw_polygon(get_id(name), points, color); }

    void tile_draw_circle(tile id, const mathpls::vec4& color) { board->tile_draw_circle(id, color); }
    void tile_draw_circle(const std::string& name, const mathpls::vec4& color) { board->tile_draw_circle(get_id(name), color); }

    void tile_draw_rectangle(tile id, float w_h, const mathpls::vec4& color) { board->tile_draw_rectangle(id, w_h, color); }
    void tile_draw_rectangle(const std::string& name, float w_h, const mathpls::vec4& color) { board->tile_draw_rectangle(get_id(name), w_h, color); }

private:
    std::unique_ptr<draw_board> board;

    size_t max_tile_size;

    detail::dense_set<tile> ids;
    detail::dense_set<std::string> names;

    std::vector<tile> auto_delete_tiles;
};

}

#undef STR
#undef XSTR

#endif //SERENA_HPP
