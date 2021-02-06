#ifndef VECTOR
#define VECTOR

#include <iostream>
#include <cmath>
#include "types.hxx"
#include <limits>

#define EPSILON 0.0000001

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_HOST __host__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_HOST
#endif 

class Vector
{
  public:
    Real x, y ,z;

    // Default constructor
    CUDA_CALLABLE_MEMBER Vector(){}

    // Constructor from three real numbers
    CUDA_CALLABLE_MEMBER Vector(Real x0, Real y0, Real z0){
      this->x = x0; this->y = y0; this->z = z0;
    }

    // Operators
    CUDA_CALLABLE_MEMBER Vector operator+( const Vector& rhs ) const {
      return Vector( x + rhs.x, y + rhs.y, z + rhs.z );
    }

    CUDA_CALLABLE_MEMBER Vector& operator+=( const Vector& rhs ) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
    }

    CUDA_CALLABLE_MEMBER Vector operator-( const Vector& rhs ) const {
      return Vector( x - rhs.x, y - rhs.y, z - rhs.z );
    }

    CUDA_CALLABLE_MEMBER Vector& operator-=( const Vector& rhs ) {
      x -= rhs.x;
      y -= rhs.y;
      z -= rhs.z;
      return *this;
    }

    CUDA_CALLABLE_MEMBER Vector operator*( Real s ) const {
      return Vector( x * s, y * s, z * s );
    }

    CUDA_CALLABLE_MEMBER Vector& operator*=( Real s ) {
      x *= s;
      y *= s;
      z *= s;
      return *this;
    }

    CUDA_CALLABLE_MEMBER Vector operator/( Real s ) const {
      Real inv = 1.0 / s;
      return Vector( x * inv, y * inv, z * inv );
    }

    CUDA_CALLABLE_MEMBER Vector& operator/=( Real s ) {
      Real inv = 1.0 / s;
      x *= inv;
      y *= inv;
      z *= inv;
      return *this;
    }

    CUDA_CALLABLE_MEMBER Vector operator-() const {
      return Vector( -x, -y, -z );
    }

    CUDA_CALLABLE_MEMBER bool operator==( const Vector& rhs ) const {
      return std::abs(x - rhs.x)<EPSILON && std::abs(y - rhs.y)<EPSILON && std::abs(z -rhs.z)<EPSILON;
    }

    CUDA_CALLABLE_MEMBER bool operator!=( const Vector& rhs ) const {
      return !operator==( rhs );
    }

    CUDA_CALLABLE_MEMBER Real norm() {
      return sqrt(x * x + y * y + z * z);
    }

    CUDA_CALLABLE_MEMBER Vector normalized(){
      double inorm = 1./this->norm();
      return Vector(x*inorm,y*inorm,z*inorm);
    }

    CUDA_CALLABLE_MEMBER void normalize(){
      double inorm = 1./this->norm();
      x*=inorm;y*=inorm;z*=inorm;
    }

};

CUDA_HOST Vector& Zeros();
CUDA_CALLABLE_MEMBER Vector operator*( Real s, Vector &u);
CUDA_HOST std::ostream &operator<< (std::ostream &stream, const Vector & u);

#endif
