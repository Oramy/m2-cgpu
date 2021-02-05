#ifndef PARSER
#define PARSER

// Type converions from string
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

// Storing options
#include <map>

class Value
{
  private:
    std::string content;

  public:
    Value() : content("") {}

    Value(std::string content_) : content(content_) {}

    double asDouble(double def=0) const
    {
      if (content == "") return def;
      return static_cast<double>(atof(content.c_str()));
    }

    int asInt(int def=0) const
    {
      if (content == "") return def;
      return atoi(content.c_str());
    }

    bool asBool(bool def=false) const
    {
      if (content == "") return def;
      if (content == "0") return false;
      if (content == "false") return false;

      return true;
    }

    std::string asString(std::string def="") const
    {
      if (content == "") return def;

      return content;
    }
};

class ArgumentParser
{
  private:
    std::map<std::string,Value> mapArguments;

  public:

    Value operator()(const std::string arg)
    {
      return mapArguments[arg];
    }

    ArgumentParser() : mapArguments(){}

    template <typename T>
      void addOption(const std::string arg, const T val) {
        std::string value = (std::ostringstream() << val).str();
        mapArguments[arg] = Value(value);
      }

    void setOptions(int argc, char **argv ){
      for (int i=1; i<argc; i++) {
        if (argv[i][0] == '-') {
          std::string values = "";
          std::string option = argv[i];
          option.erase(std::remove(option.begin(), option.end(), '-'), option.end());

          if (i < argc-1) {
            // If next argument is an option then break
            if (argv[i+1][0] == '-') {
              break;
            }
            values += argv[i+1];
          }
          if (mapArguments.find(option) == mapArguments.end()) {
            usage();
            exit (8);
          }
          // Update value in map
          mapArguments[option] = Value(values);
          i += 1;
        }
        else {
          usage();
          exit (8);
        }
      }
      std::cout<<"The simulation will be executed with the following parameters "<<std::endl;
      for (std::map<std::string,Value>::const_iterator it = mapArguments.begin(); it != mapArguments.end(); ++it) {
        std::cout << "[-"<< it->first << " "<< it->second.asString() <<" ] ";
      }
      std::cout<<std::endl;
    }

    void usage(void) const {
      std::cout<<"Usage: ./boids ";
      for (std::map<std::string,Value>::const_iterator it = mapArguments.begin(); it != mapArguments.end(); ++it) {
        std::cout << "[-"<<it->first << " val] ";
      }
      std::cout<<std::endl;
    }
};

#endif
