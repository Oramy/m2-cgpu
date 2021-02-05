#ifndef VECTOR
#define VECTOR

#include <iostream>
#include <cmath>
#include "types.hxx"
#include <limits>

#define EPSILON std::numeric_limits<Real>::epsilon()

class Vector
{
  public:
    Real x, y ,z;

    // Default constructor
    Vector(){}

    // Constructor from three real numbers
    Vector(Real x0, Real y0, Real z0){
      this->x = x0; this->y = y0; this->z = z0;
    }

    // Operators
    Vector operator+( const Vector& rhs ) const {
      return Vector( x + rhs.x, y + rhs.y, z + rhs.z );
    }

    Vector& operator+=( const Vector& rhs ) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
    }

    Vector operator-( const Vector& rhs ) const {
      return Vector( x - rhs.x, y - rhs.y, z - rhs.z );
    }

    Vector& operator-=( const Vector& rhs ) {
      x -= rhs.x;
      y -= rhs.y;
      z -= rhs.z;
      return *this;
    }

    Vector operator*( Real s ) const {
      return Vector( x * s, y * s, z * s );
    }

    Vector& operator*=( Real s ) {
      x *= s;
      y *= s;
      z *= s;
      return *this;
    }

    Vector operator/( Real s ) const {
      Real inv = 1.0 / s;
      return Vector( x * inv, y * inv, z * inv );
    }

    Vector& operator/=( Real s ) {
      Real inv = 1.0 / s;
      x *= inv;
      y *= inv;
      z *= inv;
      return *this;
    }

    Vector operator-() const {
      return Vector( -x, -y, -z );
    }

    bool operator==( const Vector& rhs ) const {
      return std::abs(x - rhs.x)<EPSILON && std::abs(y - rhs.y)<EPSILON && std::abs(z -rhs.z)<EPSILON;
    }

    bool operator!=( const Vector& rhs ) const {
      return !operator==( rhs );
    }

    Real norm() {
      return sqrt(x * x + y * y + z * z);
    }

    Vector normalized(){
      double inorm = 1./this->norm();
      return Vector(x*inorm,y*inorm,z*inorm);
    }

    void normalize(){
      double inorm = 1./this->norm();
      x*=inorm;y*=inorm;z*=inorm;
    }

};

Vector& Zeros();
Vector operator*( Real s, Vector &u);
std::ostream &operator<< (std::ostream &stream, const Vector & u);

#endif
