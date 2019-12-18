/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef MATRIX_MARKET_HPP_
#define MATRIX_MARKET_HPP_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <limits.h>

template< typename ScalarType, typename Offset, typename OrdinalType>
Offset SparseMatrix_WriteBinaryFormat(const char* filename, OrdinalType& nrows, OrdinalType& ncols, Offset& nnz,
                                    ScalarType*& values, Offset*& rowPtr, OrdinalType*& colInd, bool sort, OrdinalType idx_offset = 0)
{
  std::string base_filename(filename);
  std::string filename_row = base_filename + "_row";
  std::string filename_col = base_filename + "_col";
  std::string filename_vals = base_filename + "_vals";
  std::string filename_descr = base_filename + "_descr";
  FILE* RowFile = fopen(filename_row.c_str(),"w");
  FILE* ColFile = fopen(filename_col.c_str(),"w");
  FILE* ValsFile = fopen(filename_vals.c_str(),"w");
  FILE* DescrFile = fopen(filename_descr.c_str(),"w");

  FILE* file = fopen(filename,"r");
  char line[512];
  line[0]='%';
  int count=-1;
  //char* symmetric = NULL;
  //int nlines;

  while(line[0]=='%')
  {
          fgets(line,511,file);
          line[511] = 0;
          count++;
          //if(count==0) symmetric=strstr(line,"symmetric");

          if(line[0]=='%')
            fprintf ( DescrFile , "%s",line);
          else
            fprintf ( DescrFile , "%i %i %i\n",(int) nrows, (int) ncols, (int) nnz);
  }
  fprintf ( DescrFile , "\n");

  //Always read/write binary format using double for scalars and int for rowptrs/colinds
  //This means the same binary files will still work even if the template parameters change.
  for(Ordinal i = 0; i < nrows + 1; i++)
  {
    int r = rowPtr[i];
    fwrite(&r, sizeof(int), 1, RowFile);
  }
  for(Offset i = 0; i < nnz; i++)
  {
    int c = colInd[i];
    fwrite(&c, sizeof(int), 1, ColFile);
    double v = values[i];
    fwrite(&v, sizeof(double), 1, ValsFile);
  }
  for(Offset i = 0; i < nnz; i++)
  {
    int c = colInd[i];
    fwrite(&c, sizeof(int), 1, ColFile);
  }

  fclose(RowFile);
  fclose(ColFile);
  fclose(ValsFile);
  fclose(DescrFile);

  Ordinal min_span = nrows+1;
  Ordinal max_span = 0;
  Ordinal ave_span = 0;
  for(Ordinal row=0; row<nrows;row++) {
    Ordinal min = nrows+1;
    Ordinal max = 0;
    for(Offset i=rowPtr[row]; i<rowPtr[row+1]; i++) {
      if(colInd[i]<min) min = colInd[i];
      if(colInd[i]>max) max = colInd[i];
    }
    if(rowPtr[row+1]>rowPtr[row]) {
      size_t span = max-min;
      if(span<min_span) min_span = span;
      if(span>max_span) max_span = span;
      ave_span += span;
    } else min_span = 0;
  }
  printf("%zu Spans: %i %i %i\n", (size_t) nnz, (int) min_span, (int) max_span, (int) (ave_span/nrows));

  return nnz;
}

template< typename ScalarType, typename OrdinalType>
int SparseMatrix_ReadBinaryFormat(const char* filename, OrdinalType &nrows, OrdinalType &ncols, OrdinalType &nnz, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  std::string base_filename(filename);
  std::string filename_row = base_filename + "_row";
  std::string filename_col = base_filename + "_col";
  std::string filename_vals = base_filename + "_vals";
  std::string filename_descr = base_filename + "_descr";
  FILE* RowFile = fopen(filename_row.c_str(),"rb");
  FILE* ColFile = fopen(filename_col.c_str(),"rb");
  FILE* ValsFile = fopen(filename_vals.c_str(),"rb");
  char line[512];
  line[0]='%';
  int count=-1;
  char* symmetric = NULL;
  //int nlines;

  while(line[0]=='%')
  {
          fgets(line,511,file);
          count++;
          if(count==0) symmetric=strstr(line,"symmetric");
  }
  rewind(file);
  for(int i=0;i<count;i++)
          fgets(line,511,file);
  fscanf(file,"%i",&nrows);
  fscanf(file,"%i",&ncols);
  fscanf(file,"%i",&nnz);
  printf("Matrix dimension: %i %i %i %s\n",nrows,ncols,nnz,symmetric?"Symmetric":"General");

  fclose(file);

  bool read_values = false; 
  if(ValsFile == NULL) 
    read_values = false;

  values = new ScalarType[nnz];
  rowPtr = new OrdinalType[nrows+1];
  colInd = new OrdinalType[nnz];

  fread ( rowPtr, sizeof(OrdinalType), nrows+1, RowFile);
  fread ( colInd, sizeof(OrdinalType), nnz, ColFile);
  
  if(read_values)

  fclose(RowFile);
  fclose(ColFile);
  if(read_values) {
    fread ( values, sizeof(ScalarType), nnz, ValsFile);
    fclose(ValsFile);
  } else {
    for(int i = 0; i<nnz; i++) 
      values[i] = 0.001*(rand()%1000);
  }

  size_t min_span = nrows+1;
  size_t max_span = 0;
  size_t ave_span = 0;
  for(int row=0; row<nrows;row++) {
    int min = nrows+1; int max = 0;
    for(int i=rowPtr[row]; i<rowPtr[row+1]; i++) {
      if(colInd[i]<min) min = colInd[i];
      if(colInd[i]>max) max = colInd[i];
    }
    if(rowPtr[row+1]>rowPtr[row]) {
      size_t span = max-min;
      if(span<min_span) min_span = span;
      if(span>max_span) max_span = span;
      ave_span += span;
    } else min_span = 0;
  }
  printf("%lu Spans: %lu %lu %lu\n",(size_t) nnz,min_span,max_span,ave_span/nrows);


  return nnz;
}

#endif /* MATRIX_MARKET_HPP_ */

