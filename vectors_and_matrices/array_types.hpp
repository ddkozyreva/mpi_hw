#ifndef ARRAY_TYPES_H
#define ARRAY_TYPES_H

#include <memory>
#include <cstdint>
#include <cstring>

using intptr_t = std::intptr_t;

template <class T>
class vec final
{
    private:
    intptr_t len;
    std::shared_ptr<T[]> data;

    public:
    vec(intptr_t n) : len(n), data(new T[n]) {};
    ~vec() = default;
    intptr_t length() {return len;}
    T* raw_ptr() {return data.get();}
    T& operator()(intptr_t idx) {return data[idx];}
};

template <class T>
class matrix final
{
    private:
    intptr_t nr_, nc_;
    std::shared_ptr<T[]> data;

    public:
    matrix(intptr_t nr, intptr_t nc) : nr_(nr), nc_(nc), data(new T[nr*nc]) {};
    ~matrix() = default;
    intptr_t length() {return nr_ * nc_;}
    intptr_t nrows() {return nr_;}
    intptr_t ncols() {return nc_;}
    T* raw_ptr() {return data.get();}
    T& operator()(intptr_t row, intptr_t col) {return data[row * nc_ + col];}
    T& operator()(intptr_t idx) {return data[idx];}
    vec<T> row(intptr_t);
    vec<T> col(intptr_t);
};

template <class T>
vec<T> matrix<T>::row(intptr_t r)
{
    vec<T> v(nc_);
    std::memcpy(v.raw_ptr(), raw_ptr() + r * nc_, nc_ * sizeof(T));
    return v;
}

template <class T>
vec<T> matrix<T>::col(intptr_t c)
{
    vec<T> v(nr_);
    for(intptr_t i=0; i < nr_; i++)
    {
        v(i) = data[i * nc_ + c];
    }
    return v;
}

#endif
