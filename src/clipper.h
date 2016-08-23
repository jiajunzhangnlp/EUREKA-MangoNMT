#ifndef CLIPPER_H
#define CLIPPER_H

namespace nplm {
  struct Clipper{
  precision_type operator() (precision_type x) const { 
    return std::min(0.5, std::max(x,-0.5));
  }
};

}

#endif


