#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
// opencv dependencies
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

const float PI = 3.1415926;
const double rho = 1.0;
const double hough_theta = PI/180.0;
const int hough_threshold = 30;
const double minLineLength = 25.0;
const double maxLineGap =30.0;

const int ref_width = 256;
const int ref_height = 256;
const float eps_1 = 0.05;
const float eps_2 = 1.0;
const float angle_thd1 = 30.0;
const float angle_thd2 = 20.0;
const float dist_thd1 = 10.0;
const float dist_thd2 = 20.0;
const float dist_thd3 = 25.0;

#define NO 0
#define YES 1 // ok
#define BUCKLING 2 //wanzhe
#define ADJUST_CAMERA 3 //tiaozheng shexiangji
#define TOO_NEAR  4 //taijin
#define TOO_FAR 5 //taiyuan


float calc_lines_angle(float *line_1,float *line_2)
{
    float x11 = line_1[0];
    float y11 = line_1[1];
    float x12 = line_1[2];
    float y12 = line_1[3];
    float x21 = line_2[0];
    float y21 = line_2[1];
    float x22 = line_2[2];
    float y22 = line_2[3];
    float theta_1 = atan2(y12-y11,x12-x11);
    float theta_2 = atan2(y22-y21,x22-x21);
    if (theta_1<0)
    { 
         theta_1 += PI;
    }
    if (theta_2<0)
    {
         theta_2 += PI;
    }

    float theta_12 = theta_2 - theta_1;
    return theta_12/PI*180.0;
}


float quad_area(float *cross)
{
     float d12 = sqrt((cross[0] - cross[2]) * (cross[0] - cross[2]) +
                         (cross[1] - cross[3]) * (cross[1] - cross[3]));
     float d23 = sqrt((cross[2] - cross[4]) * (cross[2] - cross[4]) +
                         (cross[3] - cross[5]) * (cross[3] - cross[5]));
     float d34 = sqrt((cross[4] - cross[6]) * (cross[4] - cross[6]) +
                         (cross[5] - cross[7]) * (cross[5] - cross[7]));
     float d41 = sqrt((cross[0] - cross[6]) * (cross[0] - cross[6]) +
                         (cross[1] - cross[7]) * (cross[1] - cross[7]));
     float d13 = sqrt((cross[0] - cross[4]) * (cross[0] - cross[4]) +
                         (cross[1] - cross[5]) * (cross[1] - cross[5]));
     //helen
     float p1 = (d12 + d23 + d13) / 2.0;
     float p2 = (d34 + d41 + d13) / 2.0;
     float area1 = sqrt(p1 * (p1 - d12) * (p1 - d23) * (p1 - d13));
     float area2 = sqrt(p2 * (p2 - d34) * (p2 - d41) * (p2 - d13));
     return (area1 + area2);
}


float line_line_dist(float *line_1,float *line_2,float *line_3,int lines_num)
{
     float x11 = line_1[0];
     float y11 = line_1[1];
     float x12 = line_1[2];
     float y12 = line_1[3];
     float x21 = line_2[0];
     float y21 = line_2[1];
     float x22 = line_2[2];
     float y22 = line_2[3];
     float x31 = line_3[0];
     float y31 = line_3[1];
     float x32 = line_3[2];
     float y32 = line_3[3];
     float len1 = sqrt((x12-x11)*(x12-x11)+(y12-y11)*(y12-y11));
     float len3 = sqrt((x32-x31)*(x32-x31)+(y32-y31)*(y32-y31));
     float theta_2 = atan2(y22-y21,x22-x21);
     float a2 = sin(theta_2);
     float b2 = -cos(theta_2);
     float c2 = -(a2*x21+b2*y21);
     float d11 = fabs(x11*a2+y11*b2+c2);
     float d12 = fabs(x12*a2+y12*b2+c2);
     float d31 = fabs(x31*a2+y31*b2+c2);
     float d32 = fabs(x32*a2+y32*b2+c2);
     float d1_min = d11<d12?d11:d12;
     float d1_max = d11<d12?d12:d11;
     float d3_min = d31<d32?d31:d32;
     float d3_max = d31<d32?d32:d31;
     float d1 = d1_min;
     float d3 = d3_min;
     if (len1 > d1_max) d1 = 0;
     if (len3 > d3_max) d3 = 0;    
     
     if (4==lines_num)
     {
        return  d1>d3?d3:d1;
     }
     else
     {
        return d1>d3?d1:d3;
     }
    
}



float cross_linepoint_dist(float *line,float cross_x,float cross_y)
{
     float dist = sqrt((line[2]-line[0])*(line[2]-line[0])+(line[3]-line[1])*(line[3]-line[1]));
     float len1 = sqrt((cross_x-line[0])*(cross_x-line[0])+(cross_y-line[1])*(cross_y-line[1]));
     float len2 = sqrt((cross_x-line[2])*(cross_x-line[2])+(cross_y-line[3])*(cross_y-line[3]));
     float len1_min = len1;
     float len1_max = len2;
     if (len2 < len1)
     {   
         len1_min = len2;
         len1_max = len1;
     }
     if(len1_max>dist)
     {
         return len1_min;
     }
     else
     {
         return 0;
     }
}

float buckling_area(float *line_1,float *line_2,float cross_x,float cross_y)
{
     float d1 = cross_linepoint_dist(line_1,cross_x,cross_y);
     float d2 = cross_linepoint_dist(line_2,cross_x,cross_y);
     return d1*d2;
    
}


int is_buckling(float *line_1,float *line_2,float *line_3,float *line_4,float *cross)
{
     // cross1: l1,l2
     // cross2: l2,l3
     // cross3: l3,l4
     // cross4: l4,l1
     float d2 = line_line_dist(line_1,line_2,line_3,4);
     float d3 = line_line_dist(line_2,line_3,line_4,4);
     float d4 = line_line_dist(line_3,line_4,line_1,4);
     float d1 = line_line_dist(line_4,line_1,line_2,4);
     //cout << d1 << "," << d2 << "," << d3 << "," << d4 << endl;
     if (d1>50.0 || d2>50.0 || d3>50.0 || d4>50.0)
     {
           return 2;
     }
     float area = quad_area(cross);
     float area_12 = buckling_area(line_1,line_2,cross[0],cross[1]);
     float area_23 = buckling_area(line_2,line_3,cross[2],cross[3]);
     float area_34 = buckling_area(line_3,line_4,cross[4],cross[5]); 
     float area_41 = buckling_area(line_4,line_1,cross[6],cross[7]);
     //cout << area_12/area << "," << area_23/area << "," << area_34/area << "," << area_41/area << endl;
     if (area_12/area>0.05 || area_23/area>0.05 || area_34/area>0.05 || area_41/area>0.05)
     {
          return 1;
     }
     return 0;
}


int is_real_quad(float *line_1,float *line_2,float *line_3,float *line_4,float *cross,float &score,float &area,int &is_bbox,float &dist_fac,int lines_num)
{
    int res = NO;
    for (int i=0; i<8; i++)
    {
        if (cross[i]<0 || cross[i] > ref_width-1)
        {
           return res;
        }
    }
    float x1 = cross[0];
    float y1 = cross[1];
    float x2 = cross[2];
    float y2 = cross[3];
    float x3 = cross[4];
    float y3 = cross[5];
    float x4 = cross[6];
    float y4 = cross[7];
    float theta_1 = atan2(y2-y1,x2-x1);
    float theta_2 = atan2(y3-y2,x3-x2);
    float theta_3 = atan2(y4-y3,x4-x3);
    float theta_4 = atan2(y1-y4,x1-x4);
    float l1 = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
    float l2 = sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2));
    float l3 = sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3));
    float l4 = sqrt((x1-x4)*(x1-x4)+(y1-y4)*(y1-y4));
    float l13 = (l1+l3)/2.0;
    float l24 = (l2+l4)/2.0;
    float f1 = l13/ref_width;
    float f2 = l24/ref_width;
    if (theta_1<0)
    {
         theta_1 += PI;;
    }
    if (theta_2<0)
    { 
         theta_2 += PI;
    }
    if (theta_3<0)
    { 
         theta_3 += PI;
    }
    if (theta_4<0)
    { 
         theta_4 += PI;
    }
    float d11 =  cross_linepoint_dist(line_1,cross[0],cross[1]);
    float d14 =  cross_linepoint_dist(line_4,cross[0],cross[1]);
    float d21 =  cross_linepoint_dist(line_1,cross[2],cross[3]);
    float d22 =  cross_linepoint_dist(line_2,cross[2],cross[3]);
    float d32 =  cross_linepoint_dist(line_2,cross[4],cross[5]);
    float d33 =  cross_linepoint_dist(line_3,cross[4],cross[5]);
    float d43 =  cross_linepoint_dist(line_3,cross[6],cross[7]);
    float d44 =  cross_linepoint_dist(line_4,cross[6],cross[7]);
    dist_fac = 1.0-(d11+d14+d21+d22+d32+d33+d43+d44)/8.0;   


    float angle_1 = fabs(theta_2-theta_1)/PI*180.0;
    float angle_2 = fabs(theta_3-theta_2)/PI*180.0;
    float angle_3 = fabs(theta_4-theta_3)/PI*180.0;
    float angle_4 = fabs(theta_1-theta_4)/PI*180.0;
    float angle_5 = fabs(theta_1-theta_3)/PI*180.0;
    float angle_6 = fabs(theta_2-theta_4)/PI*180.0;
    if (angle_5 > 90.0){angle_5 = 180.0 -angle_5;}
    if (angle_6 > 90.0){angle_6 = 180.0 -angle_6;}
    area = quad_area(cross);
    area /= (ref_width*ref_height);
    score = 1.0 - (fabs(angle_1-90.0)/90.0 + fabs(angle_2-90.0)/90.0 + fabs(angle_3-90.0)/90.0 +fabs(angle_4-90.0)/90.0 + angle_5/90.0+ angle_6/90.0)/6.0;
    int rotate3d = (abs(angle_1-90.0)>15.0) + (fabs(angle_2-90.0)>15.0) + (fabs(angle_3-90.0)>15.0) + (fabs(angle_4-90.0)>15.0);
    if (fabs(angle_1-90.0)<angle_thd2 && fabs(angle_2-90.0)<angle_thd2 && fabs(angle_3-90.0)<angle_thd2 && fabs(angle_4-90.0)<angle_thd2 && angle_5<angle_thd2 && angle_6<angle_thd2 && l13>dist_thd3 && l24 > dist_thd3 && f1<0.8 &&f2<0.8)
    {
         float factor = l13/l24;
         if (factor > 0.5 && factor < 2.0)
         {
              res = YES;
              is_bbox = 1;
         }
    }
    else
    {
         return NO;
    }
    int buck = 0;
    if (4==lines_num)
    {
         buck = is_buckling(line_1,line_2,line_3,line_4,cross);
    }
    if (3==lines_num)
    {
        float d2 = line_line_dist(line_1,line_2,line_3,4);
        float d3 = line_line_dist(line_2,line_3,line_4,3);
        float d1 = line_line_dist(line_4,line_1,line_2,3);
        if (d1>50.0 || d2>50.0 || d3>50.0)
        {
            buck = 2;
        }
        float area = quad_area(cross);
        float area_12 = buckling_area(line_1,line_2,cross[0],cross[1]);
        float area_23 = buckling_area(line_2,line_3,cross[2],cross[3]);
        float len4 = sqrt((line_4[2]-line_4[0])*(line_4[2]-line_4[0])+(line_4[3]-line_4[1])*(line_4[3]-line_4[1]));
        float len34 = sqrt((cross[4]-cross[6])*(cross[4]-cross[6])+(cross[5]-cross[7])*(cross[5]-cross[7]));
        float cos_theta =((line_4[2]-line_4[0])*(cross[4]-cross[6]) + (line_4[3]-line_4[1])*(cross[5]-cross[7])) / (len4*len34);
        float area4 = len4*len34*sqrt(1-cos_theta*cos_theta);
        if (area_12/area>0.05 || area_23/area>0.05 || area4>0.1)
        {
           buck = 1;
        }

    }
    if(f1 >=0.95 || f2>= 0.95)
    {
         res = TOO_NEAR;
    }
    else if (f1<0.25 || f2<0.25)
    {
         res = TOO_FAR;
    }
    else if (1==buck)
    {
         res = BUCKLING;
    }
    else if (0==buck && 2<= rotate3d)
    {
         res = ADJUST_CAMERA;
    }

    return res;

}


int is_valid_cross(float *line_1,float *line_2,float &cross_x,float &cross_y)
{
    float x11 = line_1[0];
    float y11 = line_1[1];
    float x12 = line_1[2];
    float y12 = line_1[3];
    float x21 = line_2[0];
    float y21 = line_2[1];
    float x22 = line_2[2];
    float y22 = line_2[3];
    float theta_1 = atan2(y12-y11,x12-x11);
    float theta_2 = atan2(y22-y21,x22-x21);
    float a1 = sin(theta_1);
    float b1 = -cos(theta_1);
    float c1 = -(a1*x11+b1*y11);
    float a2 = sin(theta_2);
    float b2 = -cos(theta_2);
    float c2 = -(a2*x21+b2*y21);
    float d11 = fabs(x11*a2+y11*b2+c2);
    float d12 = fabs(x12*a2+y12*b2+c2);
    float d21 = fabs(x21*a1+y21*b1+c1);
    float d22 = fabs(x22*a1+y22*b1+c1);
    
    float det = a2*b1-a1*b2;
    if (fabs(det)> 0.001)
    {  
        cross_y = (a1*c2-a2*c1)/det;
        cross_x = (b2*c1-b1*c2)/det;
    }

    if ((d11<dist_thd1 || d12<dist_thd1) && (d21<dist_thd1 || d22<dist_thd1))
    {
         return 1;
    }
   
    return 0;
}


int is_valid_pingxing(float *line_1,float *line_2,float *line_3)
{
    float x11 = line_1[0];
    float y11 = line_1[1];
    float x12 = line_1[2];
    float y12 = line_1[3];
    float x21 = line_2[0];
    float y21 = line_2[1];
    float x22 = line_2[2];
    float y22 = line_2[3];
    float x31 = line_3[0];
    float y31 = line_3[1];
    float x32 = line_3[2];
    float y32 = line_3[3];
    float theta_1 = atan2(y12-y11,x12-x11);
    float theta_2 = atan2(y22-y21,x22-x21);
    float a1 = sin(theta_1);
    float b1 = -cos(theta_1);
    float c1 = -(a1*x11+b1*y11);
    float a2 = sin(theta_2);
    float b2 = -cos(theta_2);
    float c2 = -(a2*x21+b2*y21);
    float x1 = (x11+x12)/2.0;
    float y1 = (y11+y12)/2.0;
    float x2 = (x21+x22)/2.0;
    float y2 = (y21+y22)/2.0;
    float d1 = fabs(x1*a2+y1*b2+c2);
    float d2 = fabs(x2*a1+y2*b1+c1);
    float l3 = sqrt((x32-x31)*(x32-x31)+(y32-y31)*(y32-y31)); 
    
    if (d1 >dist_thd2 && d2>dist_thd2)
    {
         return 1;
    }
    else
    {
         return 0;
    }
}


float self_dist(float *line_1,float *line_2,float &dist)
{
    float x11 = line_1[0];
    float y11 = line_1[1];
    float x12 = line_1[2];
    float y12 = line_1[3];
    float x21 = line_2[0];
    float y21 = line_2[1];
    float x22 = line_2[2];
    float y22 = line_2[3];
    float theta_1 = atan2(y12-y11,x12-x11);
    float theta_2 = atan2(y22-y21,x22-x21);
    float a1 = sin(theta_1);
    float b1 = -cos(theta_1);
    float c1 = -(a1*x11+b1*y11);
    float a2 = sin(theta_2);
    float b2 = -cos(theta_2);
    float c2 = -(a2*x21+b2*y21);
    float d11 = fabs(x11*a2+y11*b2+c2);
    float d12 = fabs(x12*a2+y12*b2+c2);
    float d21 = fabs(x21*a1+y21*b1+c1);
    float d22 = fabs(x22*a1+y22*b1+c1);
    float dist_2 = (d11+d12+d21+d22)/4.0; 
    if (theta_1 < 0)
    {
        theta_1 += PI;
    }
    if (theta_2 < 0)
    {
        theta_2 += PI;
    }
    float dist_1 = fabs(theta_1-theta_2);
    if ((dist_1 < eps_1 || dist_1 > PI-eps_1) && dist_2 < eps_2)
    {
        dist = dist_2;
        return 0;
    }
    else
    {
        return 1.0;
    }
}


void lines_cluster(float *data, int m,float **cluster,int m2,int n2,int &cluster_num, int *cluster_size)
{
    cluster_num = 0;
    for (int i=0; i<m2; i++)
    { 
         cluster_size[i] = 0;
    }
    cluster[0][0] = data[0];
    cluster[0][1] = data[1];
    cluster[0][2] = data[2];
    cluster[0][3] = data[3];
    cluster_size[0]++;
    cluster_num++;
    for (int i=1; i<m;i++)
    {
         float v[4];
         v[0] = data[4*i];
         v[1] = data[4*i+1];
         v[2] = data[4*i+2];
         v[3] = data[4*i+3];
         bool flag = true;
         float min_dist = 1000000.0;
         int index = -1;
         for (int p=0; p<cluster_num;p++)
         {
               for(int q=0; q<cluster_size[p]; q++)
               {
                   float u[4];
                   u[0] = cluster[p][4*q];
                   u[1] = cluster[p][4*q+1];
                   u[2] = cluster[p][4*q+2];
                   u[3] = cluster[p][4*q+3];
                   float real_dist = 1000000.0;
                   float tmp_d = self_dist(u,v,real_dist);
                   if (tmp_d < 0.5 && real_dist < min_dist)
                   {
                        min_dist = real_dist;
                        index = p;
                        flag = false;
                   }
               }
         }
         if (flag)
         {
              cluster[cluster_num][0] = v[0];
              cluster[cluster_num][1] = v[1];
              cluster[cluster_num][2] = v[2];
              cluster[cluster_num][3] = v[3];
              cluster_size[cluster_num]++;
              cluster_num++;
         }
         else
         {
              cluster[index][4*cluster_size[index]] = v[0];
              cluster[index][4*cluster_size[index]+1] = v[1];
              cluster[index][4*cluster_size[index]+2] = v[2];
              cluster[index][4*cluster_size[index]+3] = v[3];
              cluster_size[index]++;
         }
    }
}


float calc_min_max_vector(float *point_vec, int m,float theta, float *line)
{
    float x1 = 0; 
    float y1 = 0;
    float x2 = 0; 
    float y2 = 0; 
    if (theta < PI*0.4 && theta > -PI*0.4)
    {
          float minx = ref_width-1;
          float maxx = 0;
          for (int i=0; i<m; i++)
          {
              float x = point_vec[2*i];
              float y = point_vec[2*i+1];
              if (x <minx)
              {
                 minx = x;
                 x1 = x;
                 y1 = y;
              }
              if (x>maxx)
              {
                 maxx = x;
                 x2 = x;
                 y2 = y;
              }
          }
    }
    else
    {
          float miny = ref_height-1;
          float maxy = 0;
          for (int i=0; i<m; i++)
          {
              float x = point_vec[2*i];
              float y = point_vec[2*i+1];
              if (y <miny)
              {
                 miny = y;
                 x1 = x;
                 y1 = y;
              }
              if (y > maxy)
              {
                 maxy = y;
                 x2 = x;
                 y2 = y;
              }
          }
    }
    line[0] = x1;
    line[1] = y1;
    line[2] = x2;
    line[3] = y2;
    float dist = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
    float acc_dist = 0;
    for (int i=0; i<m/2; i++)
    {
         float x1 = point_vec[4*i];
         float y1 = point_vec[4*i+1];
         float x2 = point_vec[4*i+2];
         float y2 = point_vec[4*i+3];
         acc_dist += sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
    }
    return 1.0-fabs(dist-acc_dist)/dist;
}


int is_rectangle(float *lines, float *scores,int m,float *final_cross)
{
    float *cross = (float*)malloc(sizeof(float)*8);
    for (int i=0; i<8; i++)
    { 
        cross[i] = -1;  
    }
    float max_quad_score = 0;   
    bool flag4 = false;
    bool flag3 = false;  
    int final_res = NO;
    if (m>=4)
    {
        for(int i1=0; i1<m; i1++)
        {
            for(int i2=0; i2<m; i2++)
            {
                 for(int i3=0; i3<m; i3++)
                 {
                      for(int i4=0; i4<m; i4++)
                      {
                          float *line1 = (float*)malloc(sizeof(float)*4);
                          float *line2 = (float*)malloc(sizeof(float)*4);
                          float *line3 = (float*)malloc(sizeof(float)*4);
                          float *line4 = (float*)malloc(sizeof(float)*4);
                          float score2 = (scores[i1]+scores[i2]+scores[i3]+scores[i4])/4.0;
                          line1[0] = lines[4*i1];
                          line1[1] = lines[4*i1+1];
                          line1[2] = lines[4*i1+2];
                          line1[3] = lines[4*i1+3];
                          line2[0] = lines[4*i2];
                          line2[1] = lines[4*i2+1];
                          line2[2] = lines[4*i2+2];
                          line2[3] = lines[4*i2+3];
                          line3[0] = lines[4*i3];
                          line3[1] = lines[4*i3+1];
                          line3[2] = lines[4*i3+2];
                          line3[3] = lines[4*i3+3];
                          line4[0] = lines[4*i4];
                          line4[1] = lines[4*i4+1];
                          line4[2] = lines[4*i4+2];
                          line4[3] = lines[4*i4+3];
                          float v1 = calc_lines_angle(line1,line2);
                          float v2 = calc_lines_angle(line2,line3);
                          float v3 = calc_lines_angle(line3,line4);
                          float v4 = calc_lines_angle(line4,line1);
                          int cond = fabs(fabs(v1)-90.0)<angle_thd1 && fabs(fabs(v2)-90.0)<angle_thd1 && fabs(fabs(v3)-90.0)<angle_thd1 && fabs(fabs(v4)-90.0)<angle_thd1;  
                          if (cond)
                          {
                              
                               int v5 = is_valid_cross(line1,line2,cross[0],cross[1]);
                               int v6 = is_valid_cross(line2,line3,cross[2],cross[3]);
                               int v7 = is_valid_cross(line3,line4,cross[4],cross[5]);
                               int v8 = is_valid_cross(line4,line1,cross[6],cross[7]);
                               int v9 =  is_valid_pingxing(line1,line3,line2);
                               int v10 = is_valid_pingxing(line2,line4,line3);
                               if ((v5+v6+v7+v8)>=2 &&(v9+v10)>=2)
                               {
                                       float score1 = 0;
                                       float area = 0;
                                       int is_bbox = 0;
                                       float dist_fac = 0;
                                       int res = is_real_quad(line1,line2,line3,line4,cross,score1,area,is_bbox,dist_fac,4);
                                       
                                       if(is_bbox)
                                       {
                                          if (score2+score1+area > max_quad_score)
                                          {
                                               max_quad_score = score2+score1+area;
                                               for (int i=0; i<8; i++)
                                               {
                                                   final_cross[i] = cross[i];
                                               }
                                               final_res = res;
                                          }
                                        
                                          flag4 = true;
                                          
                                       }
                               }
                          }
                          free(line1);
                          line1 = NULL;
                          free(line2);
                          line2 = NULL;
                          free(line3);
                          line3 = NULL;
                          free(line4);
                          line4 = NULL;

                       } //i4
                  } //i3
             }//i2
         }//i1
     }//if
    

    
    if (flag4)
    {
        if (NULL != cross)
        {
           free(cross);
           cross = NULL;
        }
        return final_res;
    }
    max_quad_score = 0;  
    final_res = NO;
    
 
    if (m>=3)
    {
        for(int i1=0; i1<m; i1++)
        {
            for(int i2=0; i2<m; i2++)
            {
                 for(int i3=0; i3<m; i3++)
                 {
                          if (i1==i2 || i2==i3 || i1==i3)
                          {
                            break;
                          }
                          float *line1 = (float*)malloc(sizeof(float)*4);
                          float *line2 = (float*)malloc(sizeof(float)*4);
                          float *line3 = (float*)malloc(sizeof(float)*4);
                          float *line4 = (float*)malloc(sizeof(float)*4);
                          float score2 = (scores[i1]+scores[i2]+scores[i3])/3.0;
                          line1[0] = lines[4*i1];
                          line1[1] = lines[4*i1+1];
                          line1[2] = lines[4*i1+2];
                          line1[3] = lines[4*i1+3];
                          line2[0] = lines[4*i2];
                          line2[1] = lines[4*i2+1];
                          line2[2] = lines[4*i2+2];
                          line2[3] = lines[4*i2+3];
                          line3[0] = lines[4*i3];
                          line3[1] = lines[4*i3+1];
                          line3[2] = lines[4*i3+2];
                          line3[3] = lines[4*i3+3];
                          float v1 = calc_lines_angle(line1,line2);
                          float v2 = calc_lines_angle(line2,line3);
                          float v3 = calc_lines_angle(line1,line3);
                          int cond = fabs(fabs(v1)-90.0)<angle_thd1 && fabs(fabs(v2)-90.0)<angle_thd1 && (v3<angle_thd1 || v3 > 180.0-angle_thd1);
                          if (cond)
                          {   
                              int v4 = is_valid_cross(line1,line2,cross[0],cross[1]);
                              int v5 = is_valid_cross(line2,line3,cross[2],cross[3]);
                              int v6 =  is_valid_pingxing(line1,line3,line2);
                              if ((v4+v5)>=2 && v6>=1)
                              {
                                  float d31 = sqrt((line3[1]-cross[3])*(line3[1]-cross[3])+(line3[0]-cross[2])*(line3[0]-cross[2]));
                                  float d32 = sqrt((line3[3]-cross[3])*(line3[3]-cross[3])+(line3[2]-cross[2])*(line3[2]-cross[2]));
                                  float d11 = sqrt((line1[1]-cross[1])*(line1[1]-cross[1])+(line1[0]-cross[0])*(line1[0]-cross[0]));
                                  float d12 = sqrt((line1[3]-cross[1])*(line1[3]-cross[1])+(line1[2]-cross[0])*(line1[2]-cross[0]));
                                  float len3 = (d31 > d32 ? d31:d32);
                                  float len1 = (d11 > d12 ? d11:d12);
                                  if (len3 > len1)
                                  {
                                      if (d31 > d32)
                                      {
                                           cross[4] = line3[0];
                                           cross[5] = line3[1];
                                      }
                                      else 
                                      {
                                           cross[4] = line3[2];
                                           cross[5] = line3[3];
                                      }
                                      line4[0] = cross[4];
                                      line4[1] = cross[5];
                                      line4[2] = line1[0];
                                      line4[3] = line1[1];
                                      if (d12 > d11)
                                      {
                                         line4[2] = line1[2];
                                         line4[3] = line1[3];
                                      }

                                      float theta4 = atan2(line2[3]-line2[1],line2[2]-line2[0]);
                                      float a4 = sin(theta4);
                                      float b4 = -cos(theta4);
                                      float c4 = -(a4*cross[4]+b4*cross[5]);
                                      float theta1 = atan2(line1[3]-line1[1],line1[2]-line1[0]);
                                      float a1 = sin(theta1);
                                      float b1 = -cos(theta1);
                                      float c1 = -(a1*line1[0]+b1*line1[1]);
                                      float det = a4*b1-a1*b4;
                                      if (fabs(det)> 0.001)
                                      {
                                             cross[7] = (a1*c4-a4*c1)/det;
                                             cross[6] = (b4*c1-b1*c4)/det;
                                      }
                                  }
                                  else
                                  {
                                      if (d11 > d12)
                                      {
                                           cross[6] = line1[0];
                                           cross[7] = line1[1];
                                      }
                                      else
                                      {
                                           cross[6] = line1[2];
                                           cross[7] = line1[3];
                                      }
                                      line4[0] = cross[6];
                                      line4[1] = cross[7];
                                      line4[2] = line3[0];
                                      line4[3] = line3[1];
                                      if (d32 > d31)
                                      {
                                         line4[2] = line3[2];
                                         line4[3] = line3[3];
                                      }

                                      float theta4 = atan2(line2[3]-line2[1],line2[2]-line2[0]);
                                      float a4 = sin(theta4);
                                      float b4 = -cos(theta4);
                                      float c4 = -(a4*cross[6]+b4*cross[7]);
                                      float theta3 = atan2(line3[3]-line3[1],line3[2]-line3[0]);
                                      float a3 = sin(theta3);
                                      float b3 = -cos(theta3);
                                      float c3 = -(a3*line3[0]+b3*line3[1]);
                                      float det = a4*b3-a3*b4;
                                      if (fabs(det)> 0.001)
                                      {
                                             cross[5] = (a3*c4-a4*c3)/det;
                                             cross[4] = (b4*c3-b3*c4)/det;
                                      }
                                      

                                  }

                                 float score1 = 0;
                                 float area = 0;
                                 int is_bbox = 0;
                                 float dist_fac = 0;
                                 int res = is_real_quad(line1,line2,line3,line4,cross,score1,area,is_bbox,dist_fac,3);
                                 if(is_bbox)
                                 {
                                          if (score2+score1+area > max_quad_score)
                                          {
                                               max_quad_score = score2+score1+area;
                                               final_res = res;
                                               for (int i=0; i<8; i++)
                                               {
                                                   final_cross[i] = cross[i];
                                               }
                                          }
                                          flag3 = true;

                                 }

                              }
                          }
                          free(line1);
                          line1 = NULL;
                          free(line2);
                          line2 = NULL;
                          free(line3);
                          line3 = NULL;
                          free(line4);
                          line4 = NULL;

                  } //i3
             }//i2
         }//i1
     }//if


     if (flag3)
     {
        if (NULL != cross)
        {
           free(cross);
           cross = NULL;
        }

        return final_res;
     }

     
     if (NULL != cross)
     {
        free(cross);
        cross = NULL;
     
     }

     return final_res;
    
}
    

void hough_process(Mat src_image, Mat edge_image)
{
    vector<Vec4i> lines;
    HoughLinesP(edge_image,lines,rho,hough_theta,hough_threshold,minLineLength,maxLineGap);
    int m = lines.size();
    if (m<1)
    {
        return;
    }
    
    float *z_data = (float*)malloc(sizeof(float)*4*m);
    for (int i=0; i<m; i++)
    {
        Vec4i vec = lines[i];
        z_data[4*i] = vec[0];
        z_data[4*i+1] = vec[1];
        z_data[4*i+2] = vec[2];
        z_data[4*i+3] = vec[3];
        //line(src_image, Point(vec[0],vec[1]), Point(vec[2],vec[3]), Scalar(0, 255, 0), 3, 2);
    }
  
    int m2 = m;
    int n2 = m;
    int cluster_num = 0;
    int *cluster_size = (int*)malloc(sizeof(int)*m);
    float **cluster = new float*[m];
    for(int i=0; i <m; i++)
    {
         cluster[i] = new float[4*m];
    }
    lines_cluster(z_data, m,cluster,m2,n2,cluster_num, cluster_size);
    
   
    
    float *new_lines = (float*)malloc(sizeof(float)*cluster_num*4);
    float *cluster_scores = (float*)malloc(sizeof(float)*cluster_num);
    for (int i=0; i<cluster_num; i++)
    {
        float *point_vec = (float*)malloc(sizeof(float)*cluster_size[i]*4);
        float acc_angle = 0;
        for (int j=0; j<cluster_size[i]; j++)
        {
            float x1 = cluster[i][4*j];
            float y1 = cluster[i][4*j+1];
            float x2 = cluster[i][4*j+2];
            float y2 = cluster[i][4*j+3];
            acc_angle += atan2(y2-y1,x2-x1);
            point_vec[4*j] = x1;
            point_vec[4*j+1] = y1;
            point_vec[4*j+2] = x2;
            point_vec[4*j+3] = y2;
        
        }
        acc_angle /= float(cluster_size[i]);
        if (acc_angle < 0)
        {
           acc_angle += PI;
        }
        if (acc_angle > PI/2.0)
        {
            acc_angle = acc_angle - PI; 
        }
        float *cluster_line = (float*)malloc(sizeof(float)*4);
        cluster_scores[i] = calc_min_max_vector(point_vec,cluster_size[i]*2,acc_angle,cluster_line);
        new_lines[4*i] = cluster_line[0];
        new_lines[4*i+1] = cluster_line[1];
        new_lines[4*i+2] = cluster_line[2];
        new_lines[4*i+3] = cluster_line[3];
        line(src_image, Point(int(cluster_line[0]),int(cluster_line[1])), Point(int(cluster_line[2]),int(cluster_line[3])), Scalar(255, 0, 0), 3 ,2);
        if (NULL != cluster_line)
        {
            free(cluster_line);
            cluster_line = NULL;
        }
        if (NULL != point_vec)
        {
            free(point_vec);
            point_vec = NULL;
        }
    
    }
  
    
    
    float cross[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    int res = is_rectangle(new_lines,cluster_scores,cluster_num,cross);
    if (cross[0] >=0 && cross[1] >=0 && cross[2] >=0 && cross[3] >=0 && cross[4] >=0 && cross[5] >=0 && cross[6] >=0 && cross[7] >=0)
    {
         line(src_image, Point(int(cross[0]),int(cross[1])), Point(int(cross[2]),int(cross[3])), Scalar(0, 0, 255), 2 ,1);
         line(src_image, Point(int(cross[2]),int(cross[3])), Point(int(cross[4]),int(cross[5])), Scalar(0, 0, 255), 2 ,1);
         line(src_image, Point(int(cross[4]),int(cross[5])), Point(int(cross[6]),int(cross[7])), Scalar(0, 0, 255), 2 ,1);
         line(src_image, Point(int(cross[6]),int(cross[7])), Point(int(cross[0]),int(cross[1])), Scalar(0, 0, 255), 2 ,1);

    }
    cout << "hough line:" << m << ";" << "cluster lines:" << cluster_num << ";" << "result:" << res << endl;
    

    if(NULL != new_lines)
    {
        free(new_lines);
        new_lines = NULL;
    }
    
    if(NULL != cluster_scores)
    {
        free(cluster_scores);
        cluster_scores = NULL;
    }


    for(int i=0; i<m; i++)
    {
         delete[] cluster[i];
 
    }
    delete[] cluster;
    if (NULL != cluster_size)
    {
        free(cluster_size);
        cluster_size = NULL;
    }

    if (NULL != z_data)
    {
        free(z_data);
        z_data = NULL;
    }
    lines.clear();  
    
}



int main(int argc, char *argv[])
{
    ifstream fin("./test_image_2.txt");
    int count = 0;
    if (fin.is_open())
    {
        while(!fin.eof())
        {
            string path1,path2;
            fin >> path1 >> path2;
            cout << path1.c_str() << "," << path2 << endl;
            string dst_path = path1 + "_cluster.jpg";
            Mat src = imread(path1.c_str(),1);
            Size dsize = Size(ref_width,ref_height);
	    Mat shrink;
	    resize(src, shrink, dsize, 0, 0, INTER_LINEAR);
            Mat edge_gray = imread(path2,0);
            Mat edge;
            Canny(edge_gray, edge, 100, 200, 3);
            hough_process(shrink, edge);
            imwrite(dst_path,shrink);
            count++;
            if (count >18) return 0; 
        }
    }
    
    return 0;
}
