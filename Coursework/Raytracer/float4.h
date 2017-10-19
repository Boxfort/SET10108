#pragma once
#include <xmmintrin.h> 
#include <smmintrin.h>

typedef __m128 float4;

typedef union {
	float4 s;
	float f[4];
	unsigned int ui[4];
} _float4_union;

inline static float4 float4_create(float x, float y, float z, float w)
{
	float4 f = { x, y, z, w };
	return f;
}

inline static float4 float4_uload4(const float *arr)
{
	float4 f = _mm_loadu_ps(arr);
	return f;
}

inline static void float4_ustore4(const float4 val, float *ary) {
	_mm_storeu_ps(ary, val);
}

inline static float4 float4_add(float4 lhs, float4 rhs) {
	float4 ret = _mm_add_ps(lhs, rhs);
	return ret;
}

inline static float4 float4_sub(float4 lhs, float4 rhs) {
	float4 ret = _mm_sub_ps(lhs, rhs);
	return ret;
}

inline static float4 float4_mul(float4 lhs, float4 rhs) {
	float4 ret = _mm_mul_ps(lhs, rhs);
	return ret;
}

inline static float4 float4_div(float4 lhs, float4 rhs) {
	float4 ret = _mm_div_ps(lhs, rhs);
	return ret;
}

inline static float4 float4_madd(float4 m1, float4 m2, float4 a) {
	return float4_add(float4_mul(m1, m2), a);
}

inline static float4 float4_mul_f(float4 lhs, float v)
{
	float4 f = float4_mul(lhs, float4_create(v, v, v, v));
	return f;
}

inline static float float4_get_x(float4 s) { _float4_union u = { s }; return u.f[0]; }
inline static float float4_get_y(float4 s) { _float4_union u = { s }; return u.f[1]; }
inline static float float4_get_z(float4 s) { _float4_union u = { s }; return u.f[2]; }
inline static float float4_get_w(float4 s) { _float4_union u = { s }; return u.f[3]; }

//Returns an empty float 4, e.g. (0,0,0,0)
inline static float4 float4_zero() { return _mm_setzero_ps(); }

//Returns a float 4 populated by v, e.g. v=4, return (4,4,4,4)
inline static float4 float4_flatten(float v)
{
	float4 f = _mm_set1_ps(v);
	return f;
}

//Sets each float value to the value of x, e.g. (1,2,3,4) -> (1,1,1,1)
inline static float4 float4_flatten_x(float4 v)
{
	float4 f = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	return f;
}

//Sets each float value to the value of y, e.g. (1,2,3,4) -> (2,2,2,2)
inline static float4 float4_flatten_y(float4 v)
{
	float4 f = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	return f;
}

//Sets each float value to the value of z, e.g. (1,2,3,4) -> (3,3,3,3)
inline static float4 float4_flatten_z(float4 v)
{
	float4 f = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	return f;
}

//Sets each float value to the value of w, e.g. (1,2,3,4) -> (4,4,4,4)
inline static float4 float4_flatten_w(float4 v)
{
	float4 f = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
	return f;
}

inline static float4 float4_sqrt(float4 v)
{
	float4 f = _mm_sqrt_ps(v);
}

inline static float4 float4_sum(float4 v) 
{
	const float4 s1 = float4_add(float4_flatten_x(v), float4_flatten_y(v));
}

//Returns the reciprocral square root of v
inline static float4 float4_rsqrt(float4 v)
{
	float4 f = _mm_rsqrt_ps(v);
	const float4 half = float4_create(0.5f, 0.5f, 0.5f, 0.5f);
	const float4 three = float4_create(3.0f, 3.0f, 3.0f, 3.0f);
	f = float4_mul(float4_mul(f, half), float4_sub(three, float4_mul(f, float4_mul(v, f))));
	return f;
}

//Returns the dot product of two float4s.
inline static float4 float4_dot(float4 lhs, float4 rhs)
{
	float4 f = _mm_dp_ps(lhs, rhs, 0x7f);
	return f;
}

//Returns float4 normalised between -1 and 1
inline static float4 float4_normalise(float4 v)
{
	float4 inverse = float4_rsqrt(float4_dot(v, v));
	return float4_mul(inverse, v);
}

//Returns cross product of two float4 (essentially returns a 3d vector because 4d cross product doesnt exist)
inline static float4 float4_cross(float4 lhs, float4 rhs)
{
	const float4 lyzx = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3, 0, 2, 1));
	const float4 lzxy = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3, 1, 0, 2));

	const float4 ryzx = _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3, 0, 2, 1));
	const float4 rzxy = _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3, 1, 0, 2));

	return _mm_sub_ps(_mm_mul_ps(lyzx, rzxy), _mm_mul_ps(lzxy, ryzx));
}

/* Try using double precision fam.*/
class vec_simd 
{
public:

	float4 components;

	inline vec_simd(const vec_simd& v) : components(v.components) {}
	inline vec_simd(const float4& v) : components(v) {}
	inline vec_simd(float x = 0.0f, float y = 0.0f, float z = 0.0f) : components(float4_create(x, y, z, 0.0f)) {}

	inline float get_x() const { return float4_get_x(components); }
	inline float get_y() const { return float4_get_y(components); }
	inline float get_z() const { return float4_get_z(components); }

	inline void load(const float *ary) { components = float4_uload4(ary); }
	inline void store(float *ary) const { float4_ustore4(components, ary); }

	inline float dot(const vec_simd& other) const
	{
		return float4_get_x(float4_dot(components, other.components));
	}

	inline vec_simd normal() const
	{
		return vec_simd(float4_normalise(components));
	}

	inline vec_simd cross(const vec_simd& other) const
	{
		return vec_simd(float4_cross(components, other.components));
	}

	inline vec_simd mult(const vec_simd& other) const
	{
		return float4_mul(components, other.components);
	}

	inline vec_simd operator+(const vec_simd& other) const
	{
		return vec_simd(float4_add(components, other.components));
	}

	inline vec_simd operator-(const vec_simd& other) const
	{
		return vec_simd(float4_sub(components, other.components));
	}

	inline vec_simd operator*(float v) const
	{
		return vec_simd(float4_mul_f(components, v));
	}

	inline vec_simd operator/(const vec_simd& other) const
	{
		return vec_simd(float4_div(components, other.components));
	}
};
