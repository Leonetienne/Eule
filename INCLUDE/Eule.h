#pragma once

/*** ../Eule/Vector2.h ***/

#include <cstdlib>
#include <sstream>

namespace Eule
{
	template <typename T> class Vector3;
	template <typename T> class Vector4;

	/** Representation of a 2d vector.
	* Contains a lot of utility methods.
	*/
	template <typename T>
	class Vector2
	{
	public:
		Vector2() : x{ 0 }, y{ 0 } {}
		Vector2(T _x, T _y) : x{ _x }, y{ _y } {}
		Vector2(const Vector2<T>& other) = default;
		Vector2(Vector2<T>&& other) noexcept = default;

		//! Will compute the dot product to another Vector2
		double DotProduct(const Vector2<T>& other) const;

		//! Will compute the cross product to another Vector2
		double CrossProduct(const Vector2<T>& other) const;

		//! Will compute the square magnitude
		double SqrMagnitude() const;

		//! Will compute the magnitude
		double Magnitude() const;

		//! Will return the normalization of this vector
		[[nodiscard]] Vector2<double> Normalize() const;

		//! Will normalize this vector
		void NormalizeSelf();

		//! Will scale self.n by scalar.n
		Vector2<T> VectorScale(const Vector2<T>& scalar) const;

		//! Will lerp itself towards other by t
		void LerpSelf(const Vector2<T>& other, double t);

		//! Will return a lerp result between this and another vector
		[[nodiscard]] Vector2<double> Lerp(const Vector2<T>& other, double t) const;

		//! Will compare if two vectors are similar to a certain epsilon value
		[[nodiscard]] bool Similar(const Vector2<T>& other, double epsilon = 0.00001) const;

		//! Will convert this vector to a Vector2i
		[[nodiscard]] Vector2<int> ToInt() const;

		//! Will convert this vector to a Vector2d
		[[nodiscard]] Vector2<double> ToDouble() const;

		T& operator[](std::size_t idx);
		const T& operator[](std::size_t idx) const;

		Vector2<T> operator+(const Vector2<T>& other) const;
		void operator+=(const Vector2<T>& other);
		Vector2<T> operator-(const Vector2<T>& other) const;
		void operator-=(const Vector2<T>& other);
		Vector2<T> operator*(const T scale) const;
		void operator*=(const T scale);
		Vector2<T> operator/(const T scale) const;
		void operator/=(const T scale);
		Vector2<T> operator-() const;

		operator Vector3<T>() const; //! Conversion method
		operator Vector4<T>() const; //! Conversion method

		void operator=(const Vector2<T>& other);
		void operator=(Vector2<T>&& other) noexcept;

		bool operator==(const Vector2<T>& other) const;
		bool operator!=(const Vector2<T>& other) const;

		friend std::ostream& operator<< (std::ostream& os, const Vector2<T>& v)
		{
			return os << "[x: " << v.x << "  y: " << v.y << "]";
		}
		friend std::wostream& operator<< (std::wostream& os, const Vector2<T>& v)
		{
			return os << L"[x: " << v.x << L"  y: " << v.y << L"]";
		}

		T x;
		T y;

		// Some handy predefines
		static const Vector2<double> up;
		static const Vector2<double> down;
		static const Vector2<double> right;
		static const Vector2<double> left;
		static const Vector2<double> one;
		static const Vector2<double> zero;
	};

	typedef Vector2<int> Vector2i;
	typedef Vector2<double> Vector2d;
}

/*** ../Eule/Random.h ***/

#include <random>

namespace Eule
{
	/** Extensive random number generator
	*/
	class Random
	{
	public:
		//! Will return a random double between `0` and `1`
		static double RandomFloat();

		//! Will return a random unsigned integer.
		static unsigned int RandomUint();

		//! Will return a random integer
		static unsigned int RandomInt();

		//! Will return a random double within a range  
		//! These bounds are INCLUSIVE!
		static double RandomRange(const double min, const double max);

		//! Will return a random integer within a range. This is faster than `(int)RandomRange(x,y)`  
		//! These bounds are INCLUSIVE!
		static int RandomIntRange(const int max, const int min);

		//! Will 'roll' a dice, returning `true` \f$100 * chance\f$ percent of the time.
		static bool RandomChance(const double chance);

	private:
		//! Will initialize the random number generator
		static void InitRng();

		static std::mt19937 rng;
		static bool isRngInitialized;

		// No instanciation! >:(
		Random();
	};
}

/*** ../Eule/gcccompat.h ***/


/*
* Some intrinsic functions such as _mm_sincos_pd are not available on g++ by default (requires some specific library).
* So let's just "re"define them manually if we're on g++.
* This way the code still works, even with the other intrinsics enabled.
*/

#if (__GNUC__ && __cplusplus)
#include <immintrin.h>
#include <math.h>

inline __m256d _mm256_sincos_pd(__m256d* __cos, __m256d __vec)
{
	double vec[4];

	_mm256_storeu_pd(vec, __vec);

	// Manually calculate cosines
	*__cos = _mm256_set_pd(
		cos(vec[3]),
		cos(vec[2]),
		cos(vec[1]),
		cos(vec[0])
	);

	// Manually calculate sines
	return _mm256_set_pd(
		sin(vec[3]),
		sin(vec[2]),
		sin(vec[1]),
		sin(vec[0])
	);
}
#endif

/*** ../Eule/Math.h ***/

#include <stdexcept>

namespace Eule
{
	/** Math utility class containing basic functions.
	*/
	class Math
	{
	public:
		//! Will return the bigger of two values
		[[nodiscard]] static constexpr double Max(const double a, const double b);

		//! Will return the smaller of two values
		[[nodiscard]] static constexpr double Min(const double a, const double b);

		//! Will return `v`, but at least `min`, and at most `max`
		[[nodiscard]] static constexpr double Clamp(const double v, const double min, const double max);

		//! Will return the linear interpolation between `a` and `b` by `t`
		[[nodiscard]] static constexpr double Lerp(double a, double b, double t);

		//! Will return the absolute value of `a`
		[[nodiscard]] static constexpr double Abs(const double a);

		//! Compares two double values with a given accuracy
		[[nodiscard]] static constexpr bool Similar(const double a, const double b, const double epsilon = 0.00001);
		
		//! Will compute the actual modulo of a fraction. The % operator returns bs for n<0.
		//! May throw division-by-zero std::logic_error
		[[nodiscard]] static int Mod(const int numerator, const int denominator);

		//! Kind of like \f$sin(counter)\f$, but it oscillates over \f$[a,b]\f$ instead of \f$[-1,1]\f$, by a given speed.  
		//! Given that \f$speed = 1\f$, the result will always be `a` if `counter` is even, and `b` if `counter` is uneven.  
		//! If `counter` is a rational, the result will oscillate between `a` and `b`, like `sin()` does.  
		//! If you increase `speed`, the oscillation frequency will increase. Meaning \f$speed = 2\f$ would result in \f$counter=0.5\f$ returning `b`.
		static double Oscillate(const double a, const double b, const double counter, const double speed);

	private:
		// No instanciation! >:(
		Math();
	};



	/*     These are just the inline methods. They have to lie in the header file.     */
	/*     The more sophisticated methods are in the .cpp					           */

	constexpr inline double Math::Max(double a, double b)
	{
		return (a > b) ? a : b;
	}

	constexpr inline double Math::Min(double a, double b)
	{
		return (a < b) ? a : b;
	}

	constexpr inline double Math::Clamp(double v, double min, double max)
	{
		return Max(Min(v, max), min);
	}

	constexpr inline double Math::Lerp(double a, double b, double t)
	{
		const double it = 1.0 - t;
		return (a * it) + (b * t);
	}

	constexpr inline double Math::Abs(const double a)
	{
		return (a > 0.0) ? a : -a;
	}

	constexpr inline bool Math::Similar(const double a, const double b, const double epsilon)
	{
		return Abs(a - b) <= epsilon;
	}
}

/*** ../Eule/Matrix4x4.h ***/

#include <cstring>
#include <array>
#include <ostream>

namespace Eule
{
	template <class T>
	class Vector3;
	typedef Vector3<double> Vector3d;

	/** A matrix 4x4 class representing a 3d transformation.
	* This matrix consists of a 3x3 matrix containing scaling and rotation information, and a vector (d,h,l)
	* representing the translation.
	*
	* ```
	* myMatrix[y][x] = 3
	*
	*  X ==============>
	* Y
	* |  # # # # # # # # # # #
	* |  #   a   b   c   d   #
	* |  #                   #
	* |  #   e   f   g   h   #
	* |  #                   #
	* V  #   i   j   k   l   #
	*    #                   #
	*    #   m   n   o   p   #
	*    # # # # # # # # # # #
	*
	* ```
	*
	* Note: This class can also be used to compute regular 4x4 multiplications. Use Multiply4x4() for that.
	*/

	class Matrix4x4
	{
	public:
		Matrix4x4();
		Matrix4x4(const Matrix4x4& other);
		Matrix4x4(Matrix4x4&& other) noexcept;

		//! Array holding the matrices values
		std::array<std::array<double, 4>, 4> v;

		Matrix4x4 operator*(const Matrix4x4& other) const;
		void operator*=(const Matrix4x4& other);

		Matrix4x4 operator/(const Matrix4x4& other) const;
		void operator/=(const Matrix4x4& other);

		//! Cellwise scaling
		Matrix4x4 operator*(const double scalar) const;
		//! Cellwise scaling
		void operator*=(const double scalar);

		//! Cellwise division
		Matrix4x4 operator/(const double denominator) const;
		//! Cellwise division
		void operator/=(const double denominator);

		//! Cellwise addition
		Matrix4x4 operator+(const Matrix4x4& other) const;
		//! Cellwise addition
		void operator+=(const Matrix4x4& other);

		//! Cellwise subtraction
		Matrix4x4 operator-(const Matrix4x4& other) const;
		//! Cellwise subtraction
		void operator-=(const Matrix4x4& other);


		std::array<double, 4>& operator[](std::size_t y);
		const std::array<double, 4>& operator[](std::size_t y) const;

		void operator=(const Matrix4x4& other);
		void operator=(Matrix4x4&& other) noexcept;

		bool operator==(const Matrix4x4& other);
		bool operator==(const Matrix4x4& other) const;
		bool operator!=(const Matrix4x4& other);
		bool operator!=(const Matrix4x4& other) const;

		//! Will return d,h,l as a Vector3d(x,y,z)
		const Vector3d GetTranslationComponent() const;
		//! Will set d,h,l from a Vector3d(x,y,z)
		void SetTranslationComponent(const Vector3d& trans);

		//! Will return this Matrix4x4 with d,h,l being set to 0
		Matrix4x4 DropTranslationComponents() const;

		//! Will return the 3x3 transpose of this matrix
		Matrix4x4 Transpose3x3() const;

		//! Will return the 4x4 transpose of this matrix
		Matrix4x4 Transpose4x4() const;

		//! Will return the Matrix4x4 of an actual 4x4 multiplication. operator* only does a 3x3
		Matrix4x4 Multiply4x4(const Matrix4x4& o) const;

		//! Will return the cofactors of this matrix, by dimension n
		Matrix4x4 GetCofactors(std::size_t p, std::size_t q, std::size_t n) const;

		//! Will return the determinant, by dimension n
		double Determinant(std::size_t n) const;

		//! Will return the adjoint of this matrix, by dimension n
		Matrix4x4 Adjoint(std::size_t n) const;

		//! Will return the 3x3-inverse of this matrix.  
		//! Meaning, the 3x3 component will be inverted, and the translation component will be negated
		Matrix4x4 Inverse3x3() const;

		//! Will return the full 4x4-inverse of this matrix
		Matrix4x4 Inverse4x4() const;

		//! Will check if the 3x3-component is inversible
		bool IsInversible3x3() const;

		//! Will check if the entire matrix is inversible
		bool IsInversible4x4() const;

		//! Will compare if two matrices are similar to a certain epsilon value
		bool Similar(const Matrix4x4& other, double epsilon = 0.00001) const;

		friend std::ostream& operator<< (std::ostream& os, const Matrix4x4& m);
		friend std::wostream& operator<< (std::wostream& os, const Matrix4x4& m);

		// Shorthands
		double& a = v[0][0];
		double& b = v[0][1];
		double& c = v[0][2];
		double& d = v[0][3];
		double& e = v[1][0];
		double& f = v[1][1];
		double& g = v[1][2];
		double& h = v[1][3];
		double& i = v[2][0];
		double& j = v[2][1];
		double& k = v[2][2];
		double& l = v[2][3];
		double& m = v[3][0];
		double& n = v[3][1];
		double& o = v[3][2];
		double& p = v[3][3];
	};
}

/*** ../Eule/Vector4.h ***/

#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace Eule
{
	template <typename T> class Vector2;
	template <typename T> class Vector3;

	/** Representation of a 4d vector.
	* Contains a lot of utility methods.
	*/
	template <typename T>
	class Vector4
	{
	public:
		Vector4() : x{ 0 }, y{ 0 }, z{ 0 }, w{ 0 } {}
		Vector4(T _x, T _y, T _z, T _w) : x{ _x }, y{ _y }, z{ _z }, w{ _w } {}
		Vector4(const Vector4<T>& other) = default;
		Vector4(Vector4<T>&& other) noexcept = default;

		//! Will compute the square magnitude
		double SqrMagnitude() const;

		//! Will compute the magnitude
		double Magnitude() const;

		//! Will return the normalization of this vector
		[[nodiscard]] Vector4<double> Normalize() const;

		//! Will normalize this vector
		void NormalizeSelf();

		//! Will scale self.n by scalar.n
		[[nodiscard]] Vector4<T> VectorScale(const Vector4<T>& scalar) const;

		//! Will lerp itself towards other by t
		void LerpSelf(const Vector4<T>& other, double t);

		//! Will return a lerp result between this and another vector
		[[nodiscard]] Vector4<double> Lerp(const Vector4<T>& other, double t) const;

		//! Will compare if two vectors are similar to a certain epsilon value
		[[nodiscard]] bool Similar(const Vector4<T>& other, double epsilon = 0.00001) const;

		//! Will convert this vector to a Vector4i
		[[nodiscard]] Vector4<int> ToInt() const;

		//! Will convert this vector to a Vector4d
		[[nodiscard]] Vector4<double> ToDouble() const;

		T& operator[](std::size_t idx);
		const T& operator[](std::size_t idx) const;

		Vector4<T> operator+(const Vector4<T>& other) const;
		void operator+=(const Vector4<T>& other);
		Vector4<T> operator-(const Vector4<T>& other) const;
		void operator-=(const Vector4<T>& other);
		Vector4<T> operator*(const T scale) const;
		void operator*=(const T scale);
		Vector4<T> operator/(const T scale) const;
		void operator/=(const T scale);
		Vector4<T> operator*(const Matrix4x4& mat) const;
		void operator*=(const Matrix4x4& mat);
		Vector4<T> operator-() const;

		operator Vector2<T>() const; //! Conversion method
		operator Vector3<T>() const; //! Conversion method

		void operator=(const Vector4<T>& other);
		void operator=(Vector4<T>&& other) noexcept;

		bool operator==(const Vector4<T>& other) const;
		bool operator!=(const Vector4<T>& other) const;

		friend std::ostream& operator << (std::ostream& os, const Vector4<T>& v)
		{
			return os << "[x: " << v.x << "  y: " << v.y << "  z: " << v.z << "  w: " << v.w << "]";
		}
		friend std::wostream& operator << (std::wostream& os, const Vector4<T>& v)
		{
			return os << L"[x: " << v.x << L"  y: " << v.y << L"  z: " << v.z << L"  w: " << v.w << L"]";
		}

		T x;
		T y;
		T z;
		T w;

		// Some handy predefines
		static const Vector4<double> up;
		static const Vector4<double> down;
		static const Vector4<double> right;
		static const Vector4<double> left;
		static const Vector4<double> forward;
		static const Vector4<double> backward;
		static const Vector4<double> future;
		static const Vector4<double> past;
		static const Vector4<double> one;
		static const Vector4<double> zero;
	};

	typedef Vector4<int> Vector4i;
	typedef Vector4<double> Vector4d;
}

/*** ../Eule/Vector3.h ***/

#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace Eule
{
	template <typename T> class Vector2;
	template <typename T> class Vector4;

	/** Representation of a 3d vector.
	* Contains a lot of utility methods.
	*/
	template <typename T>
	class Vector3
	{
	public:
		Vector3() : x{ 0 }, y{ 0 }, z{ 0 } {}
		Vector3(T _x, T _y, T _z) : x{ _x }, y{ _y }, z{ _z } {}
		Vector3(const Vector3<T>& other) = default;
		Vector3(Vector3<T>&& other) noexcept = default;

		//! Will compute the dot product to another Vector3
		double DotProduct(const Vector3<T>& other) const;

		//! Will compute the cross product to another Vector3
		Vector3<double> CrossProduct(const Vector3<T>& other) const;

		//! Will compute the square magnitude
		double SqrMagnitude() const;

		//! Will compute the magnitude
		double Magnitude() const;

		//! Will return the normalization of this vector
		[[nodiscard]] Vector3<double> Normalize() const;

		//! Will normalize this vector
		void NormalizeSelf();

		//! Will scale self.n by scalar.n
		[[nodiscard]] Vector3<T> VectorScale(const Vector3<T>& scalar) const;

		//! Will lerp itself towards other by t
		void LerpSelf(const Vector3<T>& other, double t);

		//! Will return a lerp result between this and another vector
		[[nodiscard]] Vector3<double> Lerp(const Vector3<T>& other, double t) const;

		//! Will compare if two vectors are similar to a certain epsilon value
		[[nodiscard]] bool Similar(const Vector3<T>& other, double epsilon = 0.00001) const;

		//! Will convert this vector to a Vector3i
		[[nodiscard]] Vector3<int> ToInt() const;

		//! Will convert this vector to a Vector3d
		[[nodiscard]] Vector3<double> ToDouble() const;

		T& operator[](std::size_t idx);
		const T& operator[](std::size_t idx) const;

		Vector3<T> operator+(const Vector3<T>& other) const;
		void operator+=(const Vector3<T>& other);
		Vector3<T> operator-(const Vector3<T>& other) const;
		void operator-=(const Vector3<T>& other);
		Vector3<T> operator*(const T scale) const;
		void operator*=(const T scale);
		Vector3<T> operator/(const T scale) const;
		void operator/=(const T scale);
		Vector3<T> operator*(const Matrix4x4& mat) const;
		void operator*=(const Matrix4x4& mat);
		Vector3<T> operator-() const;

		operator Vector2<T>() const; //! Conversion method
		operator Vector4<T>() const; //! Conversion method

		void operator=(const Vector3<T>& other);
		void operator=(Vector3<T>&& other) noexcept;

		bool operator==(const Vector3<T>& other) const;
		bool operator!=(const Vector3<T>& other) const;

		friend std::ostream& operator << (std::ostream& os, const Vector3<T>& v)
		{
			return os << "[x: " << v.x << "  y: " << v.y << "  z: " << v.z << "]";
		}
		friend std::wostream& operator << (std::wostream& os, const Vector3<T>& v)
		{
			return os << L"[x: " << v.x << L"  y: " << v.y << L"  z: " << v.z << L"]";
		}

		T x;
		T y;
		T z;

		// Some handy predefines
		static const Vector3<double> up;
		static const Vector3<double> down;
		static const Vector3<double> right;
		static const Vector3<double> left;
		static const Vector3<double> forward;
		static const Vector3<double> backward;
		static const Vector3<double> one;
		static const Vector3<double> zero;
	};

	typedef Vector3<int> Vector3i;
	typedef Vector3<double> Vector3d;
}

/*** ../Eule/Quaternion.h ***/

#include <mutex>

namespace Eule
{
    /** 3D rotation representation
    */
    class Quaternion
    {
    public:
        Quaternion();

        //! Constructs by these raw values
        explicit Quaternion(const Vector4d values);

        //! Copies this existing Quaternion
        Quaternion(const Quaternion& q);

        //! Creates an quaternion from euler angles
        Quaternion(const Vector3d eulerAngles);

        ~Quaternion();

        //! Copies
        Quaternion operator= (const Quaternion& q);

        //! Multiplies (applies)
        Quaternion operator* (const Quaternion& q) const;

        //! Divides (applies)
        Quaternion operator/ (Quaternion& q) const;

        //! Also multiplies
        Quaternion& operator*= (const Quaternion& q);

        //! Also divides
        Quaternion& operator/= (const Quaternion& q);

        //! Will transform a 3d point around its origin
        Vector3d operator* (const Vector3d& p) const;

        bool operator== (const Quaternion& q) const;
        bool operator!= (const Quaternion& q) const;

        Quaternion Inverse() const;

        Quaternion Conjugate() const;

        Quaternion UnitQuaternion() const;

        //! Will rotate a vector by this quaternion
        Vector3d RotateVector(const Vector3d& vec) const;

        //! Will return euler angles representing this Quaternion's rotation
        Vector3d ToEulerAngles() const;

        //! Will return a rotation matrix representing this Quaternions rotation
        Matrix4x4 ToRotationMatrix() const;

        //! Will return the raw four-dimensional values
        Vector4d GetRawValues() const;

        //! Will return the value between two Quaternion's as another Quaternion
        Quaternion AngleBetween(const Quaternion& other) const;

        //! Will set the raw four-dimensional values
        void SetRawValues(const Vector4d values);

        //! Will return the lerp result between two quaternions
        Quaternion Lerp(const Quaternion& other, double t) const;

        friend std::ostream& operator<< (std::ostream& os, const Quaternion& q);
        friend std::wostream& operator<< (std::wostream& os, const Quaternion& q);

    private:
        //! Scales
        Quaternion operator* (const double scale) const;
        Quaternion& operator*= (const double scale);

        //! Quaternion values
        Vector4d v;

        //! Will force a regenartion of the euler and matrix caches on further converter calls
        void InvalidateCache();

        // Caches for conversions
        mutable bool isCacheUpToDate_euler = false;
        mutable Vector3d cache_euler;

        mutable bool isCacheUpToDate_matrix = false;
        mutable Matrix4x4 cache_matrix;

        mutable bool isCacheUpToDate_inverse = false;
        mutable Vector4d cache_inverse;

        // Mutexes for thread-safe caching
        mutable std::mutex lock_eulerCache;
        mutable std::mutex lock_matrixCache;
        mutable std::mutex lock_inverseCache;
    };
}

/*** ../Eule/Constants.h ***/


// Pretty sure the compiler will optimize these calculations out...

//! Pi up to 50 decimal places
static constexpr double PI = 3.14159265358979323846264338327950288419716939937510;

//! Pi divided by two
static constexpr double HALF_PI = 1.57079632679489661923132169163975144209858469968755;

//! Factor to convert degrees to radians 
static constexpr double Deg2Rad = 0.0174532925199432957692369076848861271344287188854172222222222222;

//! Factor to convert radians to degrees
static constexpr double Rad2Deg = 57.295779513082320876798154814105170332405472466564427711013084788;

/*** ../Eule/Collider.h ***/


namespace Eule
{
	/** Abstract class of a collider domain.
	* Specializations describe a shape in 3d space, and provide implementations of the methods below,
	* for their specific shape. Examples could be a SphereCollider, a BoxCollider, etc...
	*/
	class Collider
	{
	public:
		//! Tests, if this Collider contains a point
		virtual bool Contains(const Vector3d& point) const = 0;
	};
}

/*** ../Eule/TrapazoidalPrismCollider.h ***/

#include <array>

namespace Eule
{
	/** A collider describing a trapazoidal prism.
	* A trapazoidal prism is basically a box, but each vertex can be manipulated individually, altering
	* the angles between faces.
	* Distorting a 2d face into 3d space will result in undefined behaviour. Each face should stay flat, relative to itself. This shape is based on QUADS!
	*/
	class TrapazoidalPrismCollider : public Collider
	{
	public:
		TrapazoidalPrismCollider();
		TrapazoidalPrismCollider(const TrapazoidalPrismCollider& other) = default;
		TrapazoidalPrismCollider(TrapazoidalPrismCollider&& other) noexcept = default;
		void operator=(const TrapazoidalPrismCollider& other);
		void operator=(TrapazoidalPrismCollider&& other) noexcept;

		//! Will return a specific vertex
		const Vector3d& GetVertex(std::size_t index) const;

		//! Will set the value of a specific vertex
		void SetVertex(std::size_t index, const Vector3d value);

		//! Tests, if this Collider contains a point
		bool Contains(const Vector3d& point) const override;

		/* Vertex identifiers */
		static constexpr std::size_t BACK = 0;
		static constexpr std::size_t FRONT = 4;
		static constexpr std::size_t LEFT = 0;
		static constexpr std::size_t RIGHT = 2;
		static constexpr std::size_t BOTTOM = 0;
		static constexpr std::size_t TOP = 1;

	private:
		enum class FACE_NORMALS : std::size_t;

		//! Will calculate the vertex normals from vertices
		void GenerateNormalsFromVertices();

		//! Returns the dot product of a given point against a specific plane of the bounding box
		double FaceDot(FACE_NORMALS face, const Vector3d& point) const;

		std::array<Vector3d, 8> vertices;


		// Face normals
		enum class FACE_NORMALS : std::size_t
		{
			LEFT = 0,
			RIGHT = 1,
			FRONT = 2,
			BACK = 3,
			TOP = 4,
			BOTTOM = 5
		};
		std::array<Vector3d, 6> faceNormals;
	};
}
