#pragma once
#include <xmmintrin.h> 
#include <smmintrin.h>

typedef __m256d double4;

typedef union {
	double4 s;
	double d[4];
	unsigned int ui[4];
} _double4_union;

inline static double4 double4_create(double x, double y, double z, double w)
{
	double4 f = { x, y, z, w };
	return f;
}

inline static double4 double4_uload4(const double *arr)
{
	double4 f = _mm256_loadu_pd(arr);
	return f;
}

inline static void double4_ustore4(const double4 val, double *ary) {
	_mm256_storeu_pd(ary, val);
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

//Returns a double 4 populated by v, e.g. v=4, return (4,4,4,4)
inline static double4 double4_flatten(double v)
{
	double4 f = _mm256_set1_pd(v);
	return f;
}

//Sets each double value to the value of x, e.g. (1,2,3,4) -> (1,1,1,1)
inline static double4 double4_flatten_x(double4 v)
{
	double4 f = _mm256_set1_pd(v.m256d_f64[0]); //_mm256_set1_pd(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	return f;
}

//Sets each double value to the value of y, e.g. (1,2,3,4) -> (2,2,2,2)
inline static double4 double4_flatten_y(double4 v)
{
	double4 f = _mm256_set1_pd(v.m256d_f64[1]); // _mm256_shuffle_pd(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	return f;
}

inline static double4 double4_sqrt(double4 v)
{
	double4 f = _mm256_sqrt_pd(v);
}

inline static double4 double4_sum(double4 v)
{
	const double4 s1 = double4_add(double4_flatten_x(v), double4_flatten_y(v));
}

/*
//Returns the reciprocral square root of v
inline static double4 double4_rsqrt(double4 v)
{
	//double4 f = _mm256_rsqrt_pd(v);

	//RSQRT

	double4 a = double4_mul(v, v);
	double add = a.m256d_f64[0] + a.m256d_f64[1] + a.m256d_f64[2];
	double4 f = _mm256_set1_pd(1.0 / sqrt(add));

	const double4 half = double4_create(0.5, 0.5, 0.5, 0.5);
	const double4 three = double4_create(3.0, 3.0, 3.0, 3.0);
	f = double4_mul(double4_mul(f, half), double4_sub(three, double4_mul(f, double4_mul(v, f))));
	return f;
}
*/

//Returns the dot product of two float4s.
inline static double4 double4_dot(double4 lhs, double4 rhs)
{
	//double4 f = _mm_dp_ps(lhs, rhs, 0x7f);
	//return f;

	double4 temp = _mm256_mul_pd(lhs, rhs);
	double add = temp.m256d_f64[0] + temp.m256d_f64[1] + temp.m256d_f64[2];
	double4 f = _mm256_set1_pd(add);
	return f;
}

//Returns double4 normalised between -1 and 1
inline static double4 double4_normalise(double4 v)
{
	//double4 inverse = double4_rsqrt(double4_dot(v, v));
	//return double4_mul(inverse, v);

	double4 f = _mm256_mul_pd(v, v);
	double add = f.m256d_f64[0] + f.m256d_f64[1] + f.m256d_f64[2] + f.m256d_f64[3];
	double4 r = _mm256_mul_pd(_mm256_set1_pd((1.0 / sqrt(add))), v);

	return r;
}

//Returns cross product of two double4 (essentially returns a 3d vector because 4d cross product doesnt exist)
inline static double4 double4_cross(double4 lhs, double4 rhs)
{
	//const double4 lyzx = _mm256_shuffle_pd(lhs, lhs, _MM_SHUFFLE(3, 0, 2, 1));
	//const double4 lzxy = _mm256_shuffle_pd(lhs, lhs, _MM_SHUFFLE(3, 1, 0, 2));

	//const double4 ryzx = _mm256_shuffle_pd(rhs, rhs, _MM_SHUFFLE(3, 0, 2, 1));
	//const double4 rzxy = _mm256_shuffle_pd(rhs, rhs, _MM_SHUFFLE(3, 1, 0, 2));

	//return _mm256_sub_pd(_mm256_mul_pd(lyzx, rzxy), _mm256_mul_pd(lzxy, ryzx));

	double4 r = double4_create((lhs.m256d_f64[1] * rhs.m256d_f64[2]) - (lhs.m256d_f64[2] * rhs.m256d_f64[1]),
								(lhs.m256d_f64[2] * rhs.m256d_f64[0]) - (lhs.m256d_f64[0] * rhs.m256d_f64[2]),
								(lhs.m256d_f64[0] * rhs.m256d_f64[1]) - (lhs.m256d_f64[1] * rhs.m256d_f64[0]),
								0.0
								);

	return r;
}

/* Try using double precision fam.*/
class vec_simdouble
{
public:

	double4 components;

	inline vec_simdouble(const vec_simdouble& v) : components(v.components) {}
	inline vec_simdouble(const double4& v) : components(v) {}
	inline vec_simdouble(double x = 0.0f, double y = 0.0f, double z = 0.0f) : components(double4_create(x, y, z, 0.0f)) {}

	inline double get_x() const { return double4_get_x(components); }
	inline double get_y() const { return double4_get_y(components); }
	inline double get_z() const { return double4_get_z(components); }

	inline void load(const double *ary) { components = double4_uload4(ary); }
	inline void store(double *ary) const { double4_ustore4(components, ary); }

	inline double dot(const vec_simdouble& other) const
	{
		return double4_get_x(double4_dot(components, other.components));
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
