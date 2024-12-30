#pragma once

#include <map>
#include <vector>
#include <chrono>
#include <string>

#include "yolo/yolo.h"


void print_avaliable_devices();

void reIndexResults(std::vector<Result>& infer_results, std::map<int, std::string>& src_map, std::map<std::string, int>& tgt_map, std::string model_name);

std::map<int, std::string> readFileToMap(const std::string &filePath);

std::string replaceSuffix(const std::string& path, const std::string& oldSuffix, const std::string& newSuffix);

std::string get_logfile_name(std::string& log_dir);

std::string getFilename(const std::string& file_path, bool with_ext);

std::vector<std::string> splitString(const std::string &str, const std::string &delim);
