#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "spdlog/sinks/basic_file_sink.h"               // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "yolo/model-utils.h"
#include "yolo/yolo.h"

using namespace std;


void print_avaliable_devices(){
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		spdlog::info("supported device name : {}", availableDevices[i]);
	}
}

void reIndexResults(vector<Result>& infer_results,
                    std::map<int, string>& src_map,
                    std::map<string, int>& tgt_map,
                    std::string nick_name) {
    for (auto& result : infer_results){
        result.nick_name = nick_name;
        for (auto& bbox : result.bboxes) {
            string class_name = src_map[bbox.label];
            int tgt_label = tgt_map[class_name];
            bbox.label = tgt_label;
        }
    }
}


std::map<int, std::string> readFileToMap(const std::string &filePath) {
    std::map<int, std::string> indexToClassMap;
    std::ifstream file(filePath);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        int index;
        std::string className;
        lineStream >> index >> className;
        indexToClassMap[index] = className;
    }

    return indexToClassMap;
}


std::string replaceSuffix(const std::string& path, const std::string& oldSuffix, const std::string& newSuffix) {
    std::string newPath = path;
    std::size_t pos = newPath.rfind(oldSuffix);
    if (pos != std::string::npos && pos == newPath.length() - oldSuffix.length()) {
        newPath.replace(pos, oldSuffix.length(), newSuffix);
    }
    return newPath;
}


string get_logfile_name(string& log_dir) {
    if (access(log_dir.c_str(), 0) != F_OK)
        mkdir(log_dir.c_str(), S_IRWXU);
    DIR* pDir = opendir(log_dir.c_str());
    struct dirent* ptr;
    vector<string> files_vector;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
    int max_num = 0;
    if (files_vector.size() != 0) {
        for (auto &file : files_vector) {
            string num_str = file.substr(0, file.find("."));
            int num = std::stoi(num_str);
            if (num > max_num)
                max_num = num;
        }
        max_num += 1;
    }

	return std::to_string(max_num);
}


std::string getFilename(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;

    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

std::vector<std::string> splitString(const std::string &str, const std::string &delim){
    std::vector<std::string> val;

    std::string::size_type pos1, pos2;
    pos2 = str.find(delim);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        val.push_back(str.substr(pos1, pos2 - pos1));

        pos1 = pos2 + delim.size();
        pos2 = str.find(delim, pos1);
    }
    if (pos1 != str.length())
        val.push_back(str.substr(pos1));
    return val;
}