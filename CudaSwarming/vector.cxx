#include "vector.hxx"

CUDA_HOST Vector& Zeros()
{
  static Vector u(0.,0.,0.);
  return u;
}

CUDA_CALLABLE_MEMBER Vector operator*( Real s, Vector &u) {
  return u*s;
}
CUDA_HOST std::ostream &operator<< (std::ostream &stream, const Vector & u){
  stream<<u.x<<" "<<u.y<<" "<<u.z<<std::endl;
  return stream;
}
