#include <slu_ddefs.h>

int main()
{
  GlobalLU_t lu;
  superlu_options_t opt;
  SuperMatrix M;
  int *i;
  double *d;
  void *v;
  char *c;
  SuperLUStat_t stat;
  mem_usage_t mem;

  dgsisx(&opt,&M,i,i,i,c,d,d,&M,&M,v,*i,&M,&M,d,d,&lu,&mem,&stat,i);
  return 0;
}
