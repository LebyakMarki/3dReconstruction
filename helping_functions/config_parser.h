#ifndef AKS_PROEKT_CONFIG_PARSER_H
#define AKS_PROEKT_CONFIG_PARSER_H

#endif //AKS_PROEKT_CONFIG_PARSER_H
#include <fstream>
#include <string>
#include <map>
#include <iostream>

struct Configuration
{
    bool with_calibration;
    bool find_points;
    bool save_points;
    bool visualize;
    bool sgbm;
    bool downscale;
    bool surf;
    int maxThreadsNumber;
    double lambda;
    double sigma;
    double vis_mult;
    int minHessian;
    int preFilterCap;
    int disparityRange;
    int minDisparity;
    int uniquenessRatio;
    int windowSize;
    int smoothP1;
    int smoothP2;
    int disparityMaxDiff;
    int speckleRange;
    int speckleWindowSize;
};

Configuration read_configuration(std::ifstream &config_stream){
    Configuration conf;
    std::map<std::string, std::string> confKeywords;
    std::string tempKey;
    std::string t;
    while(config_stream >> tempKey){
        config_stream >> t;
        confKeywords.insert(std::pair<std::string, std::string>(tempKey, t));
    }
    config_stream.close();

    conf.with_calibration = (confKeywords["with_calibration:"] == "true") ? true : false;
    conf.find_points = (confKeywords["find_points:"] == "true") ? true : false;
    conf.save_points = (confKeywords["save_points:"] == "true") ? true : false;
    conf.visualize = (confKeywords["visualize:"] == "true") ? true : false;
    conf.sgbm = (confKeywords["sgbm:"] == "true") ? true : false;
    conf.downscale = (confKeywords["downscale:"] == "true") ? true : false;
    conf.surf = (confKeywords["surf:"] == "true") ? true : false;

    std::stringstream(confKeywords["maxThreadsNumber:"])>>conf.maxThreadsNumber;
    std::stringstream(confKeywords["lambda:"])>>conf.lambda;
    std::stringstream(confKeywords["sigma:"])>>conf.sigma;
    std::stringstream(confKeywords["vis_mult:"])>>conf.vis_mult;
    std::stringstream(confKeywords["minHessian:"])>>conf.minHessian;
    std::stringstream(confKeywords["preFilterCap:"])>>conf.preFilterCap;
    std::stringstream(confKeywords["disparityRange:"])>>conf.disparityRange;
    std::stringstream(confKeywords["minDisparity:"])>>conf.minDisparity;
    std::stringstream(confKeywords["uniquenessRatio:"])>>conf.uniquenessRatio;
    std::stringstream(confKeywords["windowSize:"])>>conf.windowSize;
    std::stringstream(confKeywords["smoothP1:"])>>conf.smoothP1;
    std::stringstream(confKeywords["smoothP2:"])>>conf.smoothP2;
    std::stringstream(confKeywords["disparityMaxDiff:"])>>conf.disparityMaxDiff;
    std::stringstream(confKeywords["speckleRange:"])>>conf.speckleRange;
    std::stringstream(confKeywords["speckleWindowSize:"])>>conf.speckleWindowSize;

    return conf;
}

