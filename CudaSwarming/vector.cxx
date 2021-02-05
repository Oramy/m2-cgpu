#include "vector.hxx"

Vector& Zeros()
{
  static Vector u(0.,0.,0.);
  return u;
}

Vector operator*( Real s, Vector &u) {
  return u*s;
}
std::ostream &operator<< (std::ostream &stream, const Vector & u){
  stream<<u.x<<" "<<u.y<<" "<<u.z<<std::endl;
  return stream;
}
