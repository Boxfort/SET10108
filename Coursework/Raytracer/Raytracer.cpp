#include "stdafx.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <xmmintrin.h> 

#define vec vec_simdouble

using namespace std;
using namespace std::chrono;

constexpr int MAX_DEPTH = 512; // Upper limit on recursion, increase this on systems with more stack size.
constexpr double PI = 3.14159265359;

template <class T, class Compare>
constexpr const T &clamp(const T &v, const T &lo, const T &hi, Compare comp)
{
	return assert(!comp(hi, lo)), comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}

template <class T>
constexpr const T &clamp(const T &v, const T &lo, const T &hi)
{
	return clamp(v, lo, hi, std::less<>());
}

struct vec_double
{
	double x, y, z;

	vec_double(double x = 0, double y = 0, double z = 0) noexcept
		: x(x), y(y), z(z)
	{
	}

	double get_x() const noexcept
	{
		return x;
	}

	double get_y() const noexcept
	{
		return y;
	}

	double get_z() const noexcept
	{
		return z;
	}

	vec_double operator+(const vec_double &other) const noexcept
	{
		return vec_double(x + other.x, y + other.y, z + other.z);
	}

	vec_double operator-(const vec_double &other) const noexcept
	{
		return vec_double(x - other.x, y - other.y, z - other.z);
	}

	vec_double operator*(double scale) const noexcept
	{
		return vec_double(x * scale, y * scale, z * scale);
	}

	vec_double mult(const vec_double &other) const noexcept
	{
		return vec_double(x * other.x, y * other.y, z * other.z);
	}

	vec_double normal() const noexcept
	{
		return *this * (1.0 / sqrt(x * x + y * y + z * z));
	}

	double dot(const vec_double &other) const noexcept
	{
		return x * other.x + y * other.y + z * other.z;
	}

	vec_double cross(const vec_double &other) const noexcept
	{
		return vec_double(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
	}
};

//Use __m128 instead of 3 floats?
struct vec_simd
{
	__m128 components;

	//Working.
	vec_simd(double x = 0.0f, double y = 0.0f, double z = 0.0f) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
		components = _mm_set_ps(0.0f, z, y, x);
		/*
		components.m128_f32[0] = x;
		components.m128_f32[1] = y;
		components.m128_f32[2] = z;
		components.m128_f32[3] = 0.0f;
		*/
	}

	//Working.
	vec_simd(__m128 vector) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
		components = vector;
	}

	//vec(const vec& copy)
	//{
	//components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
	//	components = copy.components;
	//}

	double get_x() const noexcept
	{
		return components.m128_f32[0];
	}

	double get_y() const noexcept
	{
		return components.m128_f32[1];
	}

	double get_z() const noexcept
	{
		return components.m128_f32[2];
	}

	//Working.
	vec_simd operator+(const vec_simd other) const noexcept
	{
		return vec_simd(_mm_add_ps(other.components, components));
	}

	vec_simd operator-(const vec_simd other) const noexcept
	{
		return vec_simd(_mm_sub_ps(other.components, components));
	}

	vec_simd operator*(double scale) const noexcept
	{
		return vec_simd(_mm_mul_ps(components, _mm_set1_ps(scale)));
	}

	vec_simd mult(const vec_simd &other) const noexcept
	{
		return vec_simd(_mm_mul_ps(other.components, components));
	}

	vec_simd normal() const noexcept
	{
		// Square each component
		__m128 result = _mm_mul_ps(components, components);
		// Sum all components
		result.m128_f32[0] = result.m128_f32[1] = result.m128_f32[2] = result.m128_f32[0] + result.m128_f32[1] + result.m128_f32[2];
		// Find reciprocal squrare root and times by original vector
		return vec_simd(_mm_mul_ps(_mm_rsqrt_ps(result), components));
	}

	double dot(const vec_simd &other) const noexcept
	{
		__m128 result = _mm_mul_ps(other.components, components);
		return result.m128_f32[0] + result.m128_f32[1] + result.m128_f32[2];
	}

	vec_simd cross(const vec_simd &other) const noexcept
	{
		return vec_simd(components.m128_f32[1] * other.components.m128_f32[2] - components.m128_f32[2] * other.components.m128_f32[1], components.m128_f32[2] * other.components.m128_f32[0] - components.m128_f32[0] * other.components.m128_f32[2], components.m128_f32[0] * other.components.m128_f32[1] - components.m128_f32[1] * other.components.m128_f32[0]);
		//return vec(_mm_sub_ps(
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 1, 0, 2))),
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 0, 2, 1)))
		//));
	}

};

struct vec_simdouble
{
	__m256d components;

	//Working.
	vec_simdouble(double x = 0.0f, double y = 0.0f, double z = 0.0f) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
		components = _mm256_set_pd(0.0f, z, y, x);
		/*
		components.m128_f32[0] = x;
		components.m128_f32[1] = y;
		components.m128_f32[2] = z;
		components.m128_f32[3] = 0.0f;
		*/

	}

	//Working.
	vec_simdouble(__m256d vector) noexcept
	{
		//components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
		components = vector;
	}

	vec_simdouble(const vec_simdouble& copy)
	{
	    //components = *(__m128*)_aligned_malloc(sizeof(double) * 4, 16);
		components = copy.components;
	}

	double get_x() const noexcept
	{
		return components.m256d_f64[0];
	}

	double get_y() const noexcept
	{
		return components.m256d_f64[1];
	}

	double get_z() const noexcept
	{
		return components.m256d_f64[2];
	}

	//Working.
	vec_simdouble operator+(const vec_simdouble other) const noexcept
	{
		return vec_simdouble(_mm256_add_pd(other.components, components));
	}

	vec_simdouble operator-(const vec_simdouble other) const noexcept
	{
		return vec_simdouble(_mm256_sub_pd(other.components, components));
	}

	vec_simdouble operator*(double scale) const noexcept
	{
		return vec_simdouble(_mm256_mul_pd(components, _mm256_set1_pd(scale)));
	}

	vec_simdouble mult(const vec_simdouble &other) const noexcept
	{
		return vec_simdouble(_mm256_mul_pd(other.components, components));
	}

	vec_simdouble normal() const noexcept
	{
		// Square each component
		__m256d result = _mm256_mul_pd(components, components);
		// Sum all components
		result.m256d_f64[0] = result.m256d_f64[1] = result.m256d_f64[2] = result.m256d_f64[0] + result.m256d_f64[1] + result.m256d_f64[2];
		// Find reciprocal squrare root and times by original vector
		return vec_simdouble(_mm256_mul_pd(_mm256_div_pd(_mm256_set1_pd(1.0f), _mm256_sqrt_pd(result)), components));
	}

	double dot(const vec_simdouble &other) const noexcept
	{
		__m256d result = _mm256_mul_pd(other.components, components);
		return (result.m256d_f64[0] + result.m256d_f64[1] + result.m256d_f64[2]);
	}

	vec_simdouble cross(const vec_simdouble &other) const noexcept
	{
		return vec_simdouble(components.m256d_f64[1] * other.components.m256d_f64[2] - components.m256d_f64[2] * other.components.m256d_f64[1], components.m256d_f64[2] * other.components.m256d_f64[0] - components.m256d_f64[0] * other.components.m256d_f64[2], components.m256d_f64[0] * other.components.m256d_f64[1] - components.m256d_f64[1] * other.components.m256d_f64[0]);
		//return vec(_mm_sub_ps(
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 1, 0, 2))),
		//	_mm_mul_ps(_mm_shuffle_ps(*other.components, *other.components, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(*components, *components, _MM_SHUFFLE(3, 0, 2, 1)))
		//));
	}

};

struct ray
{
	vec origin, direction;

	ray(const vec &origin, const vec &direction) noexcept
		: origin(origin), direction(direction)
	{
	}
};

enum struct reflection_type { DIFFUSE, SPECULAR, REFRACTIVE };

struct sphere
{
	double radius;
	vec position;
	vec emission, colour;
	reflection_type reflection;

	sphere(double radius, const vec &position, const vec &emission, const vec &colour, reflection_type reflection) noexcept
		: radius(radius), position(position), emission(emission), colour(colour), reflection(reflection)
	{
	}

	double intersection(const ray &ray) const noexcept
	{
		static constexpr double eps = 1e-4;
		vec origin_position = position - ray.origin;
		double b = origin_position.dot(ray.direction);
		double determinant = b * b - origin_position.dot(origin_position) + radius * radius;
		if (determinant < 0)
		{
			return 0;
		}
		else
		{
			determinant = sqrt(determinant);
		}
		double t = b - determinant;
		if (t > eps)
		{
			return t;
		}
		else
		{
			t = b + determinant;
			if (t > eps)
			{
				return t;
			}
			else
			{
				return 0;
			}
		}
	}
};

inline bool intersect(const vector<sphere> &spheres, const ray &ray, double &distance, int &sphere_index) noexcept
{
	static constexpr double maximum_distance = 1e20;
	distance = maximum_distance;
	for (int index = 0; index < spheres.size(); ++index)
	{
		double temp_distance = spheres[index].intersection(ray);
		if (temp_distance > 0 && temp_distance < distance)
		{
			distance = temp_distance;
			sphere_index = index;
		}
	}
	return distance < maximum_distance;
}

vec radiance(const vector<sphere> &spheres, const ray &the_ray, int depth) noexcept
{
	static random_device rd;
	static default_random_engine generator(rd());
	static uniform_real_distribution<double> distribution;
	static auto get_random_number = bind(distribution, generator);

	double distance;
	int sphere_index;
	if (!intersect(spheres, the_ray, distance, sphere_index))
		return vec();
	const sphere &hit_sphere = spheres[sphere_index];
	vec hit_point = the_ray.origin + the_ray.direction * distance;
	vec intersection_normal = (hit_point - hit_sphere.position).normal();
	vec pos_intersection_normal = intersection_normal.dot(the_ray.direction) < 0.0 ? intersection_normal : intersection_normal * -1.0;
	vec colour = hit_sphere.colour;
	double max_reflection = max({ colour.get_x() , colour.get_y() , colour.get_z() });
	if (depth > MAX_DEPTH)
	{
		if (hit_sphere.emission.get_x() != 0.0) {
			auto g = distance;
		}
		return hit_sphere.emission;
	}
	else if (++depth > 5.0)
	{
		if (get_random_number() < max_reflection)
		{
			colour = colour * (1.0 / max_reflection);
		}
		else
		{
			if (hit_sphere.emission.get_x() != 0.0) {
				auto g = distance;
			}
			return hit_sphere.emission;
		}
	}

	if (hit_sphere.reflection == reflection_type::DIFFUSE)
	{
		double r1 = 2.0 * PI * get_random_number();
		double r2 = get_random_number();
		vec w = pos_intersection_normal;
		vec u = ((abs(w.get_x()) > 0.1 ? vec(0.0, 1.0, 0.0) : vec(1.0, 0.0, 0.0)).cross(w)).normal();
		vec v = w.cross(u);
		vec new_direction = (u * cos(r1) * sqrt(r2) + v * sin(r1) * sqrt(r2) + w * sqrt(1 - r2)).normal();
		
		if (depth > 1)
			cout << "";

		vec rr = radiance(spheres, ray(hit_point, new_direction), depth);
		vec r = hit_sphere.emission + colour.mult(rr);

		return r;
	}
	else if (hit_sphere.reflection == reflection_type::SPECULAR)
	{
		//cout << sphere_index << " Emssion " << hit_sphere.emission.get_x() << " | " << hit_sphere.emission.get_y() << " | " << hit_sphere.emission.get_z() << endl;
		//cout << sphere_index << " Colour  " << colour.components.m128_f32[0] << " | " << colour.components.m128_f32[1] << " | " << colour.components.m128_f32[2] << endl;

		return hit_sphere.emission + colour.mult(radiance(spheres, ray(hit_point, the_ray.direction - intersection_normal * 2.0 * intersection_normal.dot(the_ray.direction)), depth));
	}
	ray reflection_ray(hit_point, the_ray.direction - intersection_normal * 2.0 * intersection_normal.dot(the_ray.direction));
	bool into = intersection_normal.dot(pos_intersection_normal) > 0.0;
	double nc = 1.0, nt = 1.5, nnt = into ? nc / nt : nt / nc;
	double ddn = the_ray.direction.dot(pos_intersection_normal);
	double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
	if (cos2t < 0.0)
	{
		return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth));
	}
	vec tdir = (the_ray.direction * nnt - intersection_normal * ((into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)))).normal();
	double a = nt - nc;
	double b = nt + nc;
	double R0 = a * a / (b * b);
	double c = 1.0 - (into ? -ddn : tdir.dot(intersection_normal));
	double Re = R0 + (1.0 - R0) * c * c * c * c * c;
	double Tr = 1.0 - Re;
	double P = 0.25 + 0.5 * Re;
	double RP = Re / P;
	double TP = Tr / (1.0 - P);
	if (depth > 2)
	{
		if (get_random_number() < P)
		{
			return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth) * RP);
		}
		else
		{
			return hit_sphere.emission + colour.mult(radiance(spheres, ray(hit_point, tdir), depth) * TP);
		}
	}
	else
	{
		return hit_sphere.emission + colour.mult(radiance(spheres, reflection_ray, depth) * Re + radiance(spheres, ray(hit_point, tdir), depth) * Tr);
	}
}

struct lwrite
{
	unsigned long value;
	unsigned size;

	lwrite(unsigned long value, unsigned size) noexcept
		: value(value), size(size)
	{
	}
};

inline std::ostream &operator<<(std::ostream &outs, const lwrite &v)
{
	unsigned long value = v.value;
	for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
		outs.put(static_cast<char>(value & 0xFF));
	return outs;
}

bool array2bmp(const std::string &filename, const vector<vec> &pixels, const int width, const int height)
{
	std::ofstream f(filename.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	if (!f)
	{
		return false;
	}
	// Write Bmp file headers
	const int headers_size = 14 + 40;
	const int padding_size = (4 - ((height * 3) % 4)) % 4;
	const int pixel_data_size = width * ((height * 3) + padding_size);
	f.put('B').put('M'); // bfType
						 // bfSize
	f << lwrite(headers_size + pixel_data_size, 4);
	// bfReserved1, bfReserved2
	f << lwrite(0, 2) << lwrite(0, 2);
	// bfOffBits, biSize
	f << lwrite(headers_size, 4) << lwrite(40, 4);
	// biWidth,  biHeight,  biPlanes
	f << lwrite(width, 4) << lwrite(height, 4) << lwrite(1, 2);
	// biBitCount, biCompression = BI_RGB ,biSizeImage
	f << lwrite(24, 2) << lwrite(0, 4) << lwrite(pixel_data_size, 4);
	// biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant
	f << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4) << lwrite(0, 4);
	// Write image data
	for (int x = height; x > 0; x--)
	{
		for (int y = 0; y < width; y++)
		{
			const auto &val = pixels[((x - 1) * width) + y];
			f.put(static_cast<char>(int(255.0 * val.get_z()))).put(static_cast<char>(int(255.0 * val.get_y()))).put(static_cast<char>(255.0 * val.get_x()));
		}
		if (padding_size)
		{
			f << lwrite(0, padding_size);
		}
	}
	return f.good();
}

int main(int argc, char **argv)
{
	vec_simdouble a = vec_simdouble(0.124, 0.8725, 0.25);
	vec_simdouble b = vec_simdouble(-0.56, -0.38725, 0.25);

	random_device rd;
	default_random_engine generator(rd());
	uniform_real_distribution<double> distribution;
	auto get_random_number = bind(distribution, generator);

	// *** These parameters can be manipulated in the algorithm to modify work undertaken ***
	constexpr int dimension = 256;
	int samples = 1; // Algorithm performs 4 * samples per pixel.
	vector<sphere> spheres
	{
		sphere(1e5, vec(1e5 + 1, 40.8, 81.6), vec(), vec(0.75, 0.25, 0.25), reflection_type::DIFFUSE),
		sphere(1e5, vec(-1e5 + 99, 40.8, 81.6), vec(), vec(0.25, 0.25, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, 1e5), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 40.8, -1e5 + 170), vec(), vec(), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, 1e5, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(1e5, vec(50, -1e5 + 81.6, 81.6), vec(), vec(0.75, 0.75, 0.75), reflection_type::DIFFUSE),
		sphere(16.5, vec(27, 16.5, 47), vec(), vec(1, 1, 1) * 0.999, reflection_type::SPECULAR),
		sphere(16.5, vec(73, 16.5, 78), vec(), vec(1, 1, 1) * 0.999, reflection_type::REFRACTIVE),
		sphere(600, vec(50, 681.6 - 0.27, 81.6), vec(12, 12, 12), vec(), reflection_type::DIFFUSE)
	};
	// **************************************************************************************

	ray camera(vec(50, 52, 295.6), vec(0, -0.042612, -1).normal());
	vec cx = vec(0.5135);
	vec cy = (cx.cross(camera.direction)).normal() * 0.5135;
	vec r;
	vector<vec> pixels(dimension * dimension);

	unsigned int NUM_THREADS = thread::hardware_concurrency();
	auto start = system_clock::now();

	//For each row of pixels.
	//#pragma omp parallel for num_threads(NUM_THREADS) shared(pixels) private(r)
	for (int y = 0; y < dimension; ++y)
	{
		std::cout << "Rendering " << dimension << " * " << dimension << "pixels. Samples:" << samples * 4 << " spp (" << 100.0 * y / (dimension - 1) << ")" << endl;

		// For each pixel in row
		int x;
		for (x = 0; x < dimension; ++x)
		{

			for (int sy = 0, i = (dimension - y - 1) * dimension + x; sy < 2; ++sy)
			{
				for (int sx = 0; sx < 2; ++sx)
				{
					r = vec();
					//Repeat for sample count.

					for (int s = 0; s < samples; ++s)
					{
						double r1 = 2 * get_random_number(), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double r2 = 2 * get_random_number(), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						vec direction = cx * static_cast<double>(((sx + 0.5 + dx) / 2 + x) / dimension - 0.5) + cy * static_cast<double>(((sy + 0.5 + dy) / 2 + y) / dimension - 0.5) + camera.direction;
						r = r + radiance(spheres, ray(camera.origin + direction * 140.0, direction.normal()), 0.0) * (1.0 / (double)(samples));
						auto p = direction.normal();
						auto q = p;
					}

					pixels[i] = pixels[i] + vec(clamp(r.get_x(), 0.0, 1.0), clamp(r.get_y(), 0.0, 1.0), clamp(r.get_z(), 0.0, 1.0)) * 0.25f;
				}
			}
		}
	}

	auto end = system_clock::now();
	auto total = end - start;


	std::cout << "img.bmp" << (array2bmp("img.bmp", pixels, dimension, dimension) ? " Saved\n" : " Save Failed\n");
	//std::cout << "Time taken: " << duration_cast<seconds>(total).count() << endl;
	//int a;
	//std::cin >> a;

	return 0;
}