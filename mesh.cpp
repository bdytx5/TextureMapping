// for texture mapping / general c++
#include <stdlib.h>
#include <dirent.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

// matrices library
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;

using Eigen::MatrixXd;
using Eigen::VectorXd;


// for ply reading
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <fstream>
#include <limits>
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Point_set_3<Point> Point_set;

// for simplification
#include <CGAL/grid_simplify_point_set.h>

// for meshing
#include <fstream>
#include <iostream>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Scale_space_surface_reconstruction_3.h>
#include <CGAL/Scale_space_reconstruction_3/Advancing_front_mesher.h>
#include <CGAL/Scale_space_reconstruction_3/Jet_smoother.h>
#include <CGAL/IO/read_off_points.h>
#include <CGAL/Timer.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;
typedef CGAL::Scale_space_surface_reconstruction_3<Kernel>      Reconstruction;
typedef CGAL::Scale_space_reconstruction_3::Advancing_front_mesher<Kernel>    Mesher;






typedef struct veticie {
    double x;
    double y;
    double z;
} verticie;


typedef struct imgData {
    MatrixXd k;
    MatrixXd r;
    MatrixXd t;
    std::string fileName;
}imgData;

typedef struct face {
    veticie p1;
    veticie p2;
    veticie p3;
    MatrixXd normal;
    imgData uvData;
    
} face;

typedef struct objObj {
    
    veticie p1;
    veticie p2;
    veticie p3;
    VectorXd uv1;
    VectorXd uv2;
    VectorXd uv3;
    std::string imgFile;
    
} objObj;


typedef struct objFile {
    
    std::vector<veticie> pts;
    std::vector<std::vector<double>> uvs;
    std::string imgFile;
    
} objFile;







void readOFF( std::vector<face> & , std::string );
void normalize3Vector(std::vector<double> &, std::vector<double> & );
std::vector<std::string> getFiles(std::string );
void readMUImgData(std::vector<std::string> , std::vector<imgData> & , std::string);
std::vector<std::string> cleanMetaDataFileList(std::vector<std::string> & );
void meshPointSet(Point_set );
void readImgData(std::vector<imgData> & imgs);

void writeOBJ(std::vector<objObj> & );
std::string remove_extension(const std::string& filename, std::string extension);
VectorXd computeCamNormal(MatrixXd t, MatrixXd r);






int main(int argc, char*argv[])
{

    /// if the .off file is not already provided, uncomment this to read in the input off file
//     need to read the PLY
//    std::ifstream f ("cube.ply");
//    Point_set point_set;
//    if (!f || !CGAL::read_ply_point_set (f, point_set))
//    {
//        std::cerr << "Can't read input file " << std::endl;
//    }
//    // need to simply the PLY
//   // point_set.remove(CGAL::grid_simplify_point_set(point_set, 5.0),point_set.end());
//    // need to mesh the ply
//    meshPointSet(point_set);
//
//

    
    std::ifstream ifs("smallDinoSet/data.txt");
    std::string value;
    getline(ifs, value); // skip first line
    std::cerr<<value<<std::endl;
    std::vector<imgData> imgs(std::stof(value));

    readImgData(imgs);

    // need to get all normal vectors from mesh .off file
    
    std::ifstream ifs2("dino2.off");
    int vertieciesCount;
    int facesCount;
    ifs2.ignore(256,'\n');
    ifs2>>vertieciesCount>>facesCount;
    ifs2.close();
    std::vector<face> faces(facesCount);
    readOFF(faces, "dino2.off");
    

   //  need to loop through each face, and see which faces have parallel normals with each img t^transpose vector, then calculate UV's, and populate the obj vector
    
    std::vector<objObj> objs;
    objs.resize((faces.size()));

    
    /// loop through each face and find the best png image to map to
    for(int i=0;i<faces.size(); i++){

            double theta = 361.0;
            int imgIndex = 0;

            for(int j=0;j<imgs.size();j++){   // blank
                // compute differece between the two, and find the most parralel sets ( between img and face )
                VectorXd camNormal(3);
                VectorXd faceNormal(3);
                faceNormal = faces[i].normal.col(0);
                faceNormal.normalize();
                camNormal = computeCamNormal(imgs[j].t, imgs[j].r);
                camNormal.normalize();
                float dot = camNormal.dot(faceNormal);
                float tempTheta = acos(dot) * (180.0/3.141592653589793238463);

                if(tempTheta < theta){
                    theta = tempTheta;
                    imgIndex= j;
                }
            }
        // need to link filename and face

        faces[i].uvData.fileName = imgs[imgIndex].fileName;


     /// need to do some UV mapping ( looping through each face


        MatrixXd K(3,3);
        MatrixXd T(3,4);

        MatrixXd XcHAT1(3,1);
        MatrixXd XcHAT2(3,1);
        MatrixXd XcHAT3(3,1);

        K.topLeftCorner(3,3) = imgs[imgIndex].k;
        T.topLeftCorner(3,3) = imgs[imgIndex].r;
        T.topRightCorner(3,1) = imgs[imgIndex].t;


        verticie p1 = faces[i].p1;
        verticie p2 = faces[i].p2;
        verticie p3 = faces[i].p3;

        VectorXd Xw1(4);
        VectorXd Xw2(4);
        VectorXd Xw3(4);

        Xw1<<p1.x,p1.y,p1.z,1; // homogenous veticies of the face
        Xw2<<p2.x,p2.y,p2.z,1;
        Xw3<<p3.x,p3.y,p3.z,1;
        // UV Calculation
        XcHAT1 = K*T*Xw1; // 3D image coordinates of texture
        XcHAT2 = K*T*Xw2;
        XcHAT3 = K*T*Xw3;



        VectorXd Xc1(3);
        VectorXd Xc2(3);
        VectorXd Xc3(3);

        MatrixXd invK(3,3);
        invK = K.inverse();

        Xc1 = (invK)*XcHAT1; // convert 3d coordinates to 2d by multiplying by K^-1
        Xc2 = (invK)*XcHAT2;
        Xc3 = (invK)*XcHAT3;


        VectorXd uv1(2);
        VectorXd uv2(2);
        VectorXd uv3(2);

        uv1<<Xc1(0)/Xc1(2),Xc1(1)/Xc1(2); // normilization
        uv2<<Xc2(0)/Xc2(2),Xc2(1)/Xc2(2);
        uv3<<Xc3(0)/Xc3(2),Xc3(1)/Xc3(2);


        objs[i].p1 = p1;
        objs[i].p2 = p2;
        objs[i].p3 = p3;
        objs[i].uv1 = VectorXd(2);
        objs[i].uv1<<uv1(0),uv1(1);
        objs[i].uv2 = VectorXd(2);
        objs[i].uv2<<uv2(0),uv2(1);
        objs[i].uv3 = VectorXd(2);
        objs[i].uv3<<uv3(0),uv3(1);
        objs[i].imgFile = remove_extension(faces[i].uvData.fileName, ".png"); 

    }
    
    
    
    writeOBJ(objs); // write the objs and mtls to files


        return EXIT_SUCCESS;
}








VectorXd computeCamNormal(MatrixXd t, MatrixXd r){
    //T= r|t

    // the number of columns in the matrix must equal the number of rows in the vector
    MatrixXd rt(4,4); rt = MatrixXd::Zero(4,4);
    rt.topLeftCorner(3,3) = r;
    rt.topRightCorner(3,1) = t;
    rt(3,3) = 1;
    MatrixXd invrt(4,4);
    invrt = rt.inverse(); // rt is basically the t vector, which is inversed, and then multiplied by the 2 points to calculate the camera vector

    VectorXd a(4); a = VectorXd::Zero(4); a(3) = 1; // 0 0 0 1
    VectorXd b(4); b = VectorXd::Zero(4); b(3) = 1; b(2) = 1; // p 0 0 1 1

    VectorXd ta(4);
    VectorXd tb(4);
    
    ta =  invrt * a;
    tb = invrt * b;
    
    VectorXd res(3);
    res<<(tb(0) - ta(0)), (tb(1) - ta(1)), (tb(2) - ta(2));
    
    

    
    return res;
}





void normalize3Vector(std::vector<double> & t, std::vector<double> & newVector){
    newVector.reserve(3);
    double length = sqrt((t[0]*t[0]) + (t[1]*t[1]) + (t[2]*t[2]));
    newVector[0] = t[0]/length;
    newVector[1] = t[1]/length;
    newVector[2] = t[2]/length;
}


std::vector<std::string> cleanMetaDataFileList(std::vector<std::string> & files){
    
    std::vector<std::string> txtFiles;
    auto it = files.begin();
    while (it != files.end())
    {
        if(((*it).substr((*it).find_last_of(".") + 1) == "txt")){
            txtFiles.push_back(*it);
        }
        ++it;
    }

    return txtFiles;
    
}





void readMUImgData(std::vector<std::string> files, std::vector<imgData> & imgs, std::string dir){
    

    int count = 0;
    double value;

    for(int i=0; i<files.size();i++){
        
        count = 0;
        std::ifstream ifs(dir+"/"+files[i]);
        
        imgs[i].k = MatrixXd::Zero(3,3);
        imgs[i].r = MatrixXd::Zero(3,3);
        imgs[i].t = MatrixXd::Zero(3,1);

        while (ifs>>value) {

            if(count < 9){
                if(count< 3){
                    imgs[i].k(0,count) = value;
                }else if(count < 6){
                    imgs[i].k(1, count - 3) = value;
                }else if(count < 9){
                    imgs[i].k(2, count - 6) = value;
                }
                
                count++;
                continue;
              
            }
            if(count < 18){
                if(count< 12){
                    imgs[i].r(0,count - 9) = value;
                }else if(count < 15){
                    imgs[i].r(1, count - 12) = value;
                }else if(count < 18){
                    imgs[i].k(2, count - 15) = value;
                    }
                
                count++;
                continue;
            }
            if(count < 21){
                imgs[i].t(count - 18, 0) = value;
            }
            count++;
            imgs[i].fileName = files[i];
            continue;
            
        }
        
        ifs.close();
    }
}

// website structure



///void readImgData(std::string fileName, std::vector<imgData> & imgs){  old stucture

void readImgData(std::vector<imgData> & imgs){
    


    int count = 0;
    int i = -1;

    
    std::ifstream ifs("smallDinoSet/data.txt");
    bool reading = true;
    bool tst = true;
    std::string value;
    getline(ifs, value); // skip first line
    while (ifs>>value) {
        
        
        if(value.find("png") != std::string::npos){
            // beginning of a new img data
            i++;
            count = 0;
            imgs[i].k = MatrixXd::Zero(3,3);
            imgs[i].r = MatrixXd::Zero(3,3);
            imgs[i].t = MatrixXd::Zero(3,1);
            imgs[i].fileName = value;
            continue;
        }

        double val = std::stod(value);

        
        if(count < 9){
            if(count< 3){
                imgs[i].k(0,count) = val;
            }else if(count < 6){
                imgs[i].k(1, count - 3) = val;
            }else if(count < 9){
                imgs[i].k(2, count - 6) = val;
            }
            
            count++;
            continue;

        }else if(count < 18){
            
            if(count< 12){
                imgs[i].r(0,count - 9) = val;
            }else if(count < 15){
                imgs[i].r(1, count - 12) = val;
            }else if(count < 18){
                imgs[i].r(2, count - 15) = val;
            }
            
            count++;
            continue;

            
        }else if(count < 21){
            
            imgs[i].t(count - 18, 0) = val;
            count++;

        }
        
        
        
    }
}









std::string remove_extension(const std::string& filename, std::string extension) {
    std::size_t lastindex = filename.find_last_of(".");
    std::string rawname = filename.substr(0, lastindex);
    return rawname+extension;
}



void writeOBJ(std::vector<objObj> & objs) {


//    typedef struct objObj {
//
//        veticie p1;
//        veticie p2;
//        veticie p3;
//        MatrixXf uv1;
//        MatrixXf uv2;
//        MatrixXf uv3;
//        std::string imgFile;

    // .obj structure  ->>>
    // mtlllib objNameName.mtl
    // v x y z
    // vt u v
    // g texName
    // usemtl texName
    // f pIndex/UVindex


// objFile object members
//    std::vector<veticie> pts;
//    std::vector<std::vector<float>> uvs;
//    std::string imgFile;

    // need to loop through each objobj, and find the objobj's with the same img file !!
    // we will then combine all of the objobj's into one objfile object, which will be stored
    // in the objFile hashmap, with each objFile pointer accessible through the file name


    std::unordered_map<std::string, objFile> objFiles;

    for (objObj obj : objs) {
        objFiles[obj.imgFile].pts.push_back(obj.p1);
        objFiles[obj.imgFile].pts.push_back(obj.p2);
        objFiles[obj.imgFile].pts.push_back(obj.p3);

        std::vector<double> uv1 = {obj.uv1(0), obj.uv1(1)};
        std::vector<double> uv2 = {obj.uv2(0), obj.uv2(1)};
        std::vector<double> uv3 = {obj.uv3(0), obj.uv3(1)};

        objFiles[obj.imgFile].uvs.push_back(uv1);
        objFiles[obj.imgFile].uvs.push_back(uv2);
        objFiles[obj.imgFile].uvs.push_back(uv3);
        objFiles[obj.imgFile].imgFile = obj.imgFile;
    }



    // need to write the objs to .obj files

    int objIndex = 0;
    for (auto it : objFiles) {
        std::ofstream ofs("textureResults/test" + std::to_string(objIndex) + ".obj");

        //    ofs<<std::get<1>(it).imgFile;

        ofs << "mtllib " << "test" + std::to_string(objIndex) + ".mtl\n";

        // write points
        for (int i = 0; i < get<1>(it).pts.size(); i++) {
            ofs << "v " + std::to_string(get<1>(it).pts[i].x) + " " + std::to_string(get<1>(it).pts[i].y) + " " +
                   std::to_string(get<1>(it).pts[i].z) + "\n";
        }
        ofs << "\n\n";
        // write uv's
        for (int i = 0; i < get<1>(it).uvs.size(); i++) {
            ofs << "vt " + std::to_string(get<1>(it).uvs[i][0]) + " " + std::to_string(get<1>(it).uvs[i][1]) + "\n";
        }
        ofs << "\n\n";

        ofs << std::string("g ") + std::string("test") + std::to_string(objIndex) + "\n";
        ofs << std::string("usemtl ") + std::string("test") + std::to_string(objIndex) + "\n";


        int pCount = 1;
        for (int i = 0; i < get<1>(it).uvs.size() / 3; i++) {
            ofs << std::string("f ") + std::to_string(pCount) + "/" + std::to_string(pCount);
            ofs << std::string(" ") + std::to_string(pCount + 1) + "/" + std::to_string(pCount + 1);
            ofs << std::string(" ") + std::to_string(pCount + 2) + "/" + std::to_string(pCount + 2) + "\n";
            pCount += 3;
        }

        std::ofstream ofs2("textureResults/test"+std::to_string(objIndex)+".mtl");
        ofs2<<std::string("newmtl ") + std::string("test") + std::to_string(objIndex) + "\n";
        ofs2<<std::string("Ka 0.000000 0.000000 0.000000\n") + std::string("Kd 1.000000 1.000000 1.000000\n") + std::string("Ks 0.000000 0.000000 0.000000\n") + "illum 2\n" + std::string("Ns 8.000000\n") +"map_Kd ";
        ofs2<<get<1>(it).imgFile;


        objIndex++;
    }



}







std::vector<std::string> getFiles(std::string path){

    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> files;

    if ((dir = opendir (path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            // printf ("%s\n", ent->d_name);
            files.push_back(std::string(ent->d_name));

        }
        closedir (dir);
        return files;
    } else {
        /* could not open directory */
        perror ("failed to open");
        return files;
    }


}

void readOFF( std::vector<face> & faces, std::string offFileName){
    std::ifstream ifs(offFileName);
    char c;
    int vertieciesCount;
    int facesCount;
    ifs.ignore(256,'\n');
    ifs>>vertieciesCount>>facesCount;
    ifs.ignore(256,'\n');

    std::vector<verticie> verts(vertieciesCount); // read all points into vector

    for(int i=0; i<vertieciesCount; i++){
        ifs>>verts[i].x>>verts[i].y>>verts[i].z;
    }


    for(int i=0; i<facesCount; i++){
        int cnt,i1,i2,i3;
        ifs>>cnt>>i1>>i2>>i3;

        faces[i].p1 = verts[i1];
        faces[i].p2 = verts[i2];
        faces[i].p3 = verts[i3];

        // std::cerr<<verts[i1].x<<verts[i2].x<<verts[i3].x;


//        // compute normals
        double ax,ay,az,bx,by,bz;
        ax = (faces[i].p2.x - faces[i].p1.x);
        ay = (faces[i].p2.y - faces[i].p1.y);
        az = (faces[i].p2.z - faces[i].p1.z);
        bx = faces[i].p3.x - faces[i].p1.x;
        by = (faces[i].p3.y - faces[i].p1.y);
        bz = (faces[i].p3.z - faces[i].p1.z);

        faces[i].normal = MatrixXd::Zero(3,1);
        faces[i].normal(0,0) = ( (ay * bz) - (az * by) );
        faces[i].normal(1,0) = ( (az * bx) - (ax * bz) );
        faces[i].normal(2,0) = ( (ax * by) - (ay * bx) );


    }
}





void meshPointSet(Point_set point_set){

    std::cerr<<point_set.points().size();
    std::vector<Point> points(point_set.points().size());
    for(int i = 0; i<point_set.points().size(); i++)
        points[i] = point_set.point(i);



    // core
    std::cerr << "done: " << points.size() << " points after simplification." <<std::endl;
    std::cerr << "Reconstruction ";
    CGAL::Timer t;
    t.start();
    // Construct the mesh in a scale space.
    std::cerr << "--1---";
    Reconstruction reconstruct (points.begin(), points.end());
    std::cerr << "--2---";
    reconstruct.increase_scale(1);
    std::cerr << "--3---";
    reconstruct.reconstruct_surface(Mesher(15.0));
    std::cout << "--4---"<<std::endl;
    std::cerr << "done in " << t.time() << " sec." << std::endl;
    t.reset();
    std::ofstream out ("dino.off");
    out << reconstruct;
    std::cerr << "Writing result in " << t.time() << " sec." << std::endl;
    std::cerr << "Done." << std::endl;

}












//for(int i=0; i<9;i++){
//    std::cout<<"k"<<imgs[0].k[i]<<std::endl;
//
//    }
//    for(int i=0; i<9;i++){
//        std::cout<<"r"<<imgs[0].r[i]<<std::endl;
//
//    }
//
//    for(int i=0; i<3;i++){
//        std::cout<<"t"<<imgs[0].t[i]<<std::endl;
//
//        }



//
//
//void readMUImgData(std::vector<std::string> files, std::vector<imgData> & imgs, std::string dir){
//
//
//
//    int count = 0;
//    float value;
//
//    for(int i=0; i<files.size();i++){
//
//        count = 0;
//        std::ifstream ifs(dir+"/"+files[i]);
//
//        imgs[i].k = MatrixXf::Zero(3,3);
//        imgs[i].r = MatrixXf::Zero(3,3);
//        imgs[i].t = MatrixXf::Zero(3,3);
//
//        while (ifs>>value) {
//
//            if(count < 9){
//                imgs[i].k[count] = value;
//                count++;
//                continue;
//            }
//            if(count < 18){
//                imgs[i].r[count - 9] = value;
//                count++;
//                continue;
//            }
//            if(count < 21){
//                imgs[i].t[count - 18] = value;
//                count++;
//                imgs[i].fileName = files[i];
//                continue;
//            }
//        }
//
//        ifs.close();
//    }
//}




//void readImgData(std::string , std::vector<imgData> & );




// website file structure, pre-eigen




// pre-eigen

//
//void readOFF( std::vector<face> & faces){
//    std::ifstream ifs("res.off");
//    char c;
//    int vertieciesCount;
//    int facesCount;
//    ifs.ignore(256,'\n');
//    ifs>>vertieciesCount>>facesCount;
//    ifs.ignore(256,'\n');
//
//    std::vector<verticie> verts(vertieciesCount); // read all points into vector
//
//    for(int i=0; i<vertieciesCount; i++){
//        ifs>>verts[i].x>>verts[i].y>>verts[i].z;
//    }
//
//
//    for(int i=0; i<facesCount; i++){
//        int cnt,i1,i2,i3;
//        ifs>>cnt>>i1>>i2>>i3;
//        faces[i].p1 = verts[i1];
//        faces[i].p2 = verts[i2];
//        faces[i].p3 = verts[i3];
//
//
//        //        // compute normals
//        float ax,ay,az,bx,by,bz;
//        ax = (faces[i].p2.x - faces[i].p1.x);
//        ay = (faces[i].p2.y - faces[i].p1.y);
//        az = (faces[i].p2.z - faces[i].p1.z);
//        bx = faces[i].p3.x - faces[i].p1.x;
//        by = (faces[i].p3.y - faces[i].p1.y);
//        bz = (faces[i].p3.z - faces[i].p1.z);
//
//
//
//        faces[i].normal.reserve(3);
//        faces[i].normal[0] = ( (ay * bz) - (az * by) );
//        faces[i].normal[1] = ( (az * bx) - (ax * bz) );
//        faces[i].normal[2] = ( (ax * by) - (ay * bx) );
//
//    }
//}






///// old main stuff



/////////// old code ////////////////////// for reading img data from website without eigen

///void readImgData(std::string fileName, std::vector<imgData> & imgs){  old stucture

//void readImgData(std::vector<std::string> files, std::vector<imgData> & imgs, std::string dir){
//
//
//
//
//
//
//    int totalImgs;
//    int rdCount = 0;
//    int index = -1;
//    int kCount = 0;
//    int rCount = 0;
//    int tCount = 0;
//
//    std::ifstream ifs("templeData/data.txt");
//    bool reading = true;
//    bool tst = true;
//    std::string value;
//    getline(ifs, value); // skip first line
//    while (ifs>>value) {
//
//
//        if(value.find("png") != std::string::npos){
//            // beginning of a new img data
//            index++;
//            imgs[index].k.reserve(9);
//            imgs[index].r.reserve(9);
//            imgs[index].t.reserve(3);
//            imgs[index].fileName = value;
//            continue;
//        }
//
//        if(rdCount < 9){
//            imgs[index].k[kCount] = std::stof(value);
//            kCount++;
//            rdCount++;
//            continue;
//        }else if(rdCount < 18){
//            imgs[index].r[rCount] = std::stof(value);
//            rCount++;
//            rdCount++;
//            continue;
//        }else if(rdCount < 21){
//            imgs[index].t[tCount] = std::stof(value);
//            tCount++;
//            rdCount++;
//            continue;
//        }
//
//        if(rdCount == 21){
//            // reset readCount and vector index's
//            tCount = 0;
//            rCount = 0;
//            kCount = 0;
//            rdCount = 0;
//        }
//
//
//
//    }
//}



//        for(int i=0;i<3;i++){
//            if(Xc1(i) < 0){
//                Xc1(i) = -Xc1(i);
//            }   if(Xc2(i) < 0){
//                Xc2(i) = -Xc2(i);
//            }   if(Xc3(i) < 0){
//                Xc3(i) = -Xc3(i);
//            }
//        };