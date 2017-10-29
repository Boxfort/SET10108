#pragma once
#include <xmmintrin.h> 
#include <smmintrin.h>

typedef __m256d double4;

typedef union {
	double4 s;
	__declspec(align(32)) double d[4];
	unsigned int ui[4];
} _double4_union;

inline static double4 double4_create(double x, double y, double z, double w)
{
	double4 f = { x, y, z, w };
	return f;
}

inline static double4 double4_add(double4 lhs, double4 rhs) {
	double4 ret = _mm256_add_pd(lhs, rhs);
	return ret;
}

inline static double4 double4_sub(double4 lhs, double4 rhs) {
	double4 ret = _mm256_sub_pd(lhs, rhs);
	return ret;
}

inline static double4 double4_mul(double4 lhs, double4 rhs) {
	double4 ret = _mm256_mul_pd(lhs, rhs);
	return ret;
}

inline static double4 double4_mul_f(double4 lhs, double v)
{
	double4 f = double4_mul(lhs, double4_create(v, v, v, v));
	return f;
}

inline static double double4_get_x(double4 s) { _double4_union u = { s }; return u.d[0]; }
inline static double double4_get_y(double4 s) { _double4_union u = { s }; return u.d[1]; }
inline static double double4_get_z(double4 s) { _double4_union u = { s }; return u.d[2]; }
inline static double double4_get_w(double4 s) { _double4_union u = { s }; return u.d[3]; }

//Returns an empty double 4, e.g. (0,0,0,0)
inline static double4 double4_zero() { return _mm256_setzero_pd(); }

//Returns the dot product of two float4s.
inline static double double4_dot(double4 lhs, double4 rhs)
{
	//double4 f = _mm_dp_ps(lhs, rhs, 0x7f);
	//return f;

	double4 temp = _mm256_mul_pd(lhs, rhs);
	double add = double4_get_x(temp) + double4_get_y(temp) + double4_get_z(temp);

	//double4 res = _mm256_add_pd(temp, _mm256_shuffle_pd(temp, temp, 1));
	//return double4_get_x(res);

	return add;
}

//Returns double4 normalised between -1 and 1
inline static double4 double4_normalise(double4 v)
{
	double4 f = _mm256_mul_pd(v, v);
	double add = double4_get_x(f) + double4_get_y(f) + double4_get_z(f);
	double4 r = _mm256_mul_pd(_mm256_set1_pd((1.0 / sqrt(add))), v);

	return r;
}

//Returns cross product of two double4 (essentially returns a 3d vector because 4d cross product doesnt exist)
inline static double4 double4_cross(double4 lhs, double4 rhs)
{
	//double4 r = double4_create((double4_get_y(lhs) * double4_get_z(rhs)) - (double4_get_z(lhs) * double4_get_y(rhs)),
	//	(double4_get_z(lhs) * double4_get_x(rhs)) - (double4_get_x(lhs) * double4_get_z(rhs)),
	//	(double4_get_x(lhs) * double4_get_y(rhs)) - (double4_get_y(lhs) * double4_get_x(rhs)),
	//	0.0
	//);
	//
	//return r;

	double4 c = _mm256_permute4x64_pd( _mm256_sub_pd ( _mm256_mul_pd(lhs, _mm256_permute4x64_pd(rhs, _MM_SHUFFLE(3, 0, 2, 1))),
													   _mm256_mul_pd(rhs, _mm256_permute4x64_pd(lhs, _MM_SHUFFLE(3, 0, 2, 1)))),
													   _MM_SHUFFLE(3, 0, 2, 1)
	);

	return c;
}

class vec_simdouble
{
public:

	double4 components;

	inline vec_simdouble(const vec_simdouble& v) : components(v.components) {}
	inline vec_simdouble(const double4& v) : components(v) {}
	inline vec_simdouble(double x, double y = 0.0f, double z = 0.0f) : components(double4_create(x, y, z, 0.0f)) {}
	inline vec_simdouble() : components(double4_zero()) {}

	inline double get_x() const { return double4_get_x(components); }
	inline double get_y() const { return double4_get_y(components); }
	inline double get_z() const { return double4_get_z(components); }

	inline double dot(const vec_simdouble& other) const
	{
		return double4_dot(components, other.components);
	}

	inline vec_simdouble normal() const
	{
		return vec_simdouble(double4_normalise(components));
	}

	inline vec_simdouble cross(const vec_simdouble& other) const
	{
		return vec_simdouble(double4_cross(components, other.components));
	}

	inline vec_simdouble mult(const vec_simdouble& other) const
	{
		return double4_mul(components, other.components);
	}

	inline vec_simdouble operator+(const vec_simdouble& other) const
	{
		return vec_simdouble(double4_add(components, other.components));
	}

	inline vec_simdouble operator-(const vec_simdouble& other) const
	{
		return vec_simdouble(double4_sub(components, other.components));
	}

	inline vec_simdouble operator*(double v) const
	{
		return vec_simdouble(double4_mul_f(components, v));
	}
};