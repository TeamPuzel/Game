#pragma once
#include "angle.hpp"
#include <concepts>
#include <primitive>

namespace math {
    template <typename Self> concept numeric =
        std::integral<Self> or
        std::floating_point<Self> or
        std::same_as<Self, fixed>;

    /// A row-major combined matrix and vector type.
    ///
    /// Note that the row-major design means you multiply from left to right and vectors are horizontal.
    /// The associated `Matrix::Vector` type provides the vector type which can be used with a given matrix type.
    ///
    /// Similarly there is a type alias, `Vector<T, N>`, which is just shorthand for a vector-like matrix.
    /// Generally you need not worry about remembering that the width is the primary vector size.
    ///
    /// The matrix provides it's dimensions as constants and the element type through `Matrix::Element`.
    ///
    /// TODO: This type sort of assumes floats in places so it needs more work now that it allows more element types.
    template <numeric T, const usize W, const usize H> requires (W > 0 and H > 0) class Matrix final {
        std::array<T, W * H> data;

      public:
        static constexpr auto width = W;
        static constexpr auto height = H;
        using Element = T;

        template <numeric U, const usize W2, const usize H2> requires (W2 > 0 and H2 > 0) friend class Matrix;

        constexpr auto operator [] (usize x, usize y) -> Element& { return data[x + y * W]; }
        constexpr auto operator [] (usize x, usize y) const -> Element const& { return data[x + y * W]; }

        /// Constructs a matrix from a 2d array of values.
        constexpr explicit Matrix(Element const(&values)[H][W]) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    (*this)[x, y] = values[y][x];
        }

        /// Constructs a zero initialized matrix.
        ///
        /// For an identity matrix use `Matrix::identity` instead.
        constexpr Matrix() {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    (*this)[x, y] = 0;
        }

        constexpr Matrix(Element value) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    (*this)[x, y] = value;
        }

        constexpr auto operator == (this Matrix const& self, Matrix const& other) -> bool {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    if (self[x, y] != other[x, y]) return false;
            return true;
        }

        constexpr auto operator != (this Matrix const& self, Matrix const& other) -> bool {
            return not (self == other);
        }

        /// Generic matrix multiplication.
        template <const usize W2, const usize H2>
            requires (W == H2)
        constexpr auto operator * (this Matrix const& self, Matrix<Element, W2, H2> const& other) -> Matrix<Element, W2, H> {
            Matrix<Element, W2, H> result;

            for (usize i = 0; i < H; i += 1)
                for (usize j = 0; j < W2; j += 1)
                    for (usize k = 0; k < W; k += 1)
                        result[j, i] += self[k, i] * other[j, k];

            return result;
        }

        constexpr auto operator * (this Matrix const& self, Element const& scalar) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] * scalar;

            return result;
        }

        constexpr auto hadamard(this Matrix const& self, Matrix const& other) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] * other[x, y];

            return result;
        }

        constexpr auto operator / (this Matrix const& self, Element const& scalar) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] / scalar;

            return result;
        }

        constexpr auto operator + (this Matrix const& self) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = +self[x, y];

            return result;
        }

        constexpr auto operator + (this Matrix const& self, Matrix const& other) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] + other[x, y];

            return result;
        }

        constexpr auto operator + (this Matrix const& self, Element const& scalar) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] + scalar;

            return result;
        }

        constexpr auto operator - (this Matrix const& self) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = -self[x, y];

            return result;
        }

        constexpr auto operator - (this Matrix const& self, Matrix const& other) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] - other[x, y];

            return result;
        }

        constexpr auto operator - (this Matrix const& self, Element const& scalar) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = self[x, y] - scalar;

            return result;
        }

        constexpr void operator += (this Matrix& self, Matrix const& other) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] + other[x, y];
        }

        constexpr void operator += (this Matrix& self, Element const& scalar) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] + scalar;
        }

        constexpr void operator -= (this Matrix& self, Matrix const& other) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] - other[x, y];
        }

        constexpr void operator -= (this Matrix& self, Element const& scalar) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] - scalar;
        }

        constexpr void operator *= (this Matrix& self, Element const& scalar) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] * scalar;
        }

        constexpr void operator /= (this Matrix& self, Element const& scalar) {
            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    self[x, y] = self[x, y] / scalar;
        }

        /// The rotation axis for constructing rotatation matrices.
        enum class RotationAxis { Pitch, Yaw, Roll };

        /// Creates a rotation matrix.
        static constexpr auto rotation(RotationAxis axis, Angle<Element> angle) -> Matrix requires (W == 4 and H == 4) {
            const auto a = angle;
            switch (axis) {
                case RotationAxis::Pitch: return Matrix({
                    { 1, 0, 0, 0 },
                    { 0, a.cos(), -a.sin(), 0 },
                    { 0, a.sin(),  a.cos(), 0 },
                    { 0, 0, 0, 1 }
                });
                case RotationAxis::Yaw: return Matrix({
                    {  a.cos(), 0, a.sin(), 0 },
                    { 0, 1, 0, 0 },
                    { -a.sin(), 0, a.cos(), 0 },
                    { 0, 0, 0, 1 }
                });
                case RotationAxis::Roll: return Matrix({
                    { a.cos(), -a.sin(), 0, 0 },
                    { a.sin(),  a.cos(), 0, 0 },
                    { 0, 0, 1, 0 },
                    { 0, 0, 0, 1 }
                });
            };
        }

        /// Creates a rotation matrix.
        static constexpr auto rotation(RotationAxis axis, Angle<Element> angle) -> Matrix requires (W == 3 and H == 3) {
            const auto a = angle;
            switch (axis) {
                case RotationAxis::Pitch: return Matrix({
                    { 1, 0, 0 },
                    { 0, a.cos(), -a.sin() },
                    { 0, a.sin(),  a.cos() }
                });
                case RotationAxis::Yaw: return Matrix({
                    {  a.cos(), 0, a.sin() },
                    { 0, 1, 0 },
                    { -a.sin(), 0, a.cos() }
                });
                case RotationAxis::Roll: return Matrix({
                    { a.cos(), -a.sin(), 0 },
                    { a.sin(),  a.cos(), 0 },
                    { 0, 0, 1 }
                });
            };
        }

        /// Creates a translation matrix.
        static constexpr auto translation(Element x, Element y) -> Matrix requires (W == 3 and H == 3) {
            return Matrix({
                { 1, 0, 0 },
                { 0, 1, 0 },
                { x, y, 1 }
            });
        }

        /// Creates a translation matrix.
        static constexpr auto translation(Element x, Element y, Element z) -> Matrix requires (W == 4 and H == 4) {
            return Matrix({
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { x, y, z, 1 }
            });
        }

        /// Creates a scaling matrix.
        static constexpr auto scaling(Element x, Element y, Element z) -> Matrix requires (W == 4 and H == 4) {
            return Matrix({
                { x, 0, 0, 0 },
                { 0, y, 0, 0 },
                { 0, 0, z, 0 },
                { 0, 0, 0, 1 }
            });
        }

        /// Creates a standard projection matrix.
        static constexpr auto projection(Element w, Element h, Angle<Element> fov, Element near, Element far) -> Matrix requires (W == 4 and H == 4) {
            const auto aspect = h / w;
            const auto q = far / (far - near);
            const auto f = 1 / (fov / rad(2)).tan();
            return Matrix({
                { aspect * f, 0, 0, 0 },
                { 0, f, 0, 0 },
                { 0, 0, q, 1 },
                { 0, 0, -near * q, 0 }
            });
        }

        /// Creates and identity matrix.
        static constexpr auto identity() -> Matrix requires (W == 4 and H == 4) {
            return Matrix({
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 }
            });
        }

        constexpr auto inverse() const -> Matrix requires (W == H) {
            Matrix result = identity();
            Matrix copy = *this;

            // Perform Gauss-Jordan elimination
            for (usize i = 0; i < W; i += 1) {
                // Find pivot
                Element pivot = copy[i, i];
                if (pivot == 0) {
                    return { std::numeric_limits<Element>::quiet_NaN() };
                }

                // Normalize the pivot row
                for (usize j = 0; j < W; j += 1) {
                    copy[i, j] /= pivot;
                    result[i, j] /= pivot;
                }

                // Eliminate other rows
                for (usize k = 0; k < W; k += 1) {
                    if (k == i) continue;
                    Element factor = copy[k, i];
                    for (usize j = 0; j < W; j += 1) {
                        copy[k, j] -= factor * copy[i, j];
                        result[k, j] -= factor * result[i, j];
                    }
                }
            }

            return result;
        }

        // Vector implementation ---------------------------------------------------------------------------------------

        /// Associated vector type.
        using Vector = Matrix<Element, W, 1>;

        static constexpr bool is_vector = H == 1;

        constexpr auto operator [] (usize index) -> Element& requires (H == 1) {
            return data[index];
        }
        constexpr auto operator [] (usize index) const -> Element const& requires (H == 1) {
            return data[index];
        }

        /// Vector constructor
        constexpr explicit Matrix(Element const(&values)[W]) requires (H == 1) {
            for (usize i = 0; i < W; i += 1)
                (*this)[i] = values[i];
        }

        /// Vector constructor
        template <std::convertible_to<Element>... Ts>
        constexpr explicit(false) Matrix(Ts... elements) requires (H == 1 and W == sizeof...(Ts)) {
            const Element values[] = { elements... };
            for (usize i = 0; i < W; i += 1)
                (*this)[i] = values[i];
        }

        template <const usize W2>
        constexpr Matrix(Matrix<T, W2, 1> truncating) requires (W < W2) {
            for (usize i = 0; i < W; i += 1)
                (*this)[i] = truncating[i];
        }

        template <const usize W2, std::convertible_to<Element>... Ts>
        constexpr Matrix(Matrix<T, W2, 1> extending, Ts... rest) requires (W == W2 + sizeof...(Ts)) {
            for (usize i = 0; i < W; i += 1)
                (*this)[i] = extending[i];

            Element r[] = { static_cast<T>(rest)... };
            for (usize i = 0; i < sizeof...(Ts); i += 1)
                (*this)[W2 + i] = r[i];
        }

        [[nodiscard]] auto x() & -> Element& requires (H == 1 and W >= 1) { return (*this)[0]; }
        [[nodiscard]] auto x() const& -> Element const& requires (H == 1 and W >= 1) { return (*this)[0]; }

        [[nodiscard]] auto y() & -> Element& requires (H == 1 and W >= 2) { return (*this)[1]; }
        [[nodiscard]] auto y() const& -> Element const& requires (H == 1 and W >= 2) { return (*this)[1]; }

        [[nodiscard]] auto z() & -> Element& requires (H == 1 and W >= 3) { return (*this)[2]; }
        [[nodiscard]] auto z() const& -> Element const& requires (H == 1 and W >= 3) { return (*this)[2]; }

        [[nodiscard]] auto w() & -> Element& requires (H == 1 and W >= 4) { return (*this)[3]; }
        [[nodiscard]] auto w() const& -> Element const& requires (H == 1 and W >= 4) { return (*this)[3]; }

        [[nodiscard]] auto r() & -> Element& requires (H == 1 and W >= 1) { return (*this)[0]; }
        [[nodiscard]] auto r() const& -> Element const& requires (H == 1 and W >= 1) { return (*this)[0]; }

        [[nodiscard]] auto g() & -> Element& requires (H == 1 and W >= 2) { return (*this)[1]; }
        [[nodiscard]] auto g() const& -> Element const& requires (H == 1 and W >= 2) { return (*this)[1]; }

        [[nodiscard]] auto b() & -> Element& requires (H == 1 and W >= 3) { return (*this)[2]; }
        [[nodiscard]] auto b() const& -> Element const& requires (H == 1 and W >= 3) { return (*this)[2]; }

        [[nodiscard]] auto a() & -> Element& requires (H == 1 and W >= 4) { return (*this)[3]; }
        [[nodiscard]] auto a() const& -> Element const& requires (H == 1 and W >= 4) { return (*this)[3]; }

        constexpr auto dot(this Vector const& self, Vector const& other) -> Element {
            Element acc = 0; for (usize i = 0; i < W; i += 1) acc += self[i] * other[i]; return acc;
        }

        constexpr auto cross(this Vector const& self, Vector const& other) -> Vector requires (W == 3) {
            return Vector {
    			self.y() * other.z() - self.z() * other.y(),
    			self.z() * other.x() - self.x() * other.z(),
    			self.x() * other.y() - self.y() * other.x()
    		};
        }

        constexpr auto magnitude(this Vector const& self) -> Element {
            Element acc = 0; for (usize i = 0; i < W; i += 1) acc += self[i] * self[i]; return std::sqrt(acc);
        }

        constexpr auto normalized(this Vector const& self) -> Vector {
            Element m = self.magnitude();
            Vector result; for (usize i = 0; i < W; i += 1) result[i] = self[i] / m; return result;
        }

        constexpr auto map(this Matrix const& self, auto&& fn) -> Matrix {
            Matrix result;

            for (usize x = 0; x < W; x += 1)
                for (usize y = 0; y < H; y += 1)
                    result[x, y] = fn(self[x, y]);

            return result;
        }
    };

    /// A vector type, an alias of a matrix.
    template <numeric T, const usize LENGTH> using Vector = Matrix<T, LENGTH, 1>;

    template <numeric... T> Matrix(T...) -> Matrix<T...[0], sizeof...(T), 1>;

    template <numeric T, const usize W, const usize H>
    constexpr auto mix(Matrix<T, W, H> lhs, Matrix<T, W, H> rhs, f32 t) -> Matrix<T, W, H> {
        Matrix<T, W, H> result;

        for (usize x = 0; x < W; x += 1)
            for (usize y = 0; y < H; y += 1)
                result[x, y] = lhs[x, y] + t * (rhs[x, y] - lhs[x, y]);

        return result;
    }

    template <std::floating_point T> constexpr auto mix(T lhs, T rhs, f32 t) -> T {
        return lhs + t * (rhs - lhs);
    }

    template <std::floating_point T> constexpr auto normalize(
        const T n,
        const T min_from, const T max_from,
        const T min_to,   const T max_to
    ) -> T {
        return (max_to - min_to) / (max_from - min_from) * (n - max_from) + max_to;
    }
}
