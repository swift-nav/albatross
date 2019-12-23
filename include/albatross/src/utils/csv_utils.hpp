/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CSV_UTILS_H
#define ALBATROSS_CSV_UTILS_H

#include <cereal/archives/xml.hpp>
#include <cereal/external/rapidxml/rapidxml.hpp>

/*
 * This contains some tools which facilitate writing datasets and
 * corresponding predicitons to CSV which can be useful for using
 * external tools to investigate a model's performance.
 *
 * There are two options for making the write_to_csv work for a
 * custom FeatureType.
 *
 * 1) Add the appropriate cereal serialization routines.  For example,
 *
 *   template <typename Archive> void serialize(Archive &archive, const
 * std::uint32_t) {
 *       archive(cereal::make_nvp("foo", foo));
 *       archive(cereal::make_nvp("bar", bar));
 *   }
 *
 *   by using `cereal::make_nvp` each variable you archive will be
 *   given a name and that name will end up being the column name
 *   in the resulting CSV.  Doing this also enables a variety of
 *   other serialization methods, including JSON, XML and BINARY.
 *
 *   This approach is likely the easier but slower option since the
 *   CSV will be made by first serializing to XML, then flattening the
 *   XML to determine the CSV columns and values.
 *
 * 2) Write your own custom `to_map` function for the specific
 *   FeatureType you want to serialize.  This allows you to
 *   derive or modify new columns in the resulting CSV.
 *
 *   auto to_map(const CustomFeature &feature) {
 *     std::map<std::string, std::string> output;
 *     output["foo"] = std::to_string(feature.foo);
 *     output["bar"] = std::to_string(feature.bar);
 *     output["other"] = std::to_string(feature.one) +
 * std::to_string(feature.two);
 *     return output;
 *   };
 *
 */

namespace albatross {

using namespace cereal;

/*
 * This recursive function generates a map from key to value in an
 * XML document by joining nested structure using a ".".
 */
template <typename NodeType>
inline std::map<std::string, std::string>
flatten_xml_serialization(const rapidxml::xml_node<NodeType> *node,
                          bool prepend_parent = false) {
  std::map<std::string, std::string> output;
  std::string value = node->value();

  if (value.size() > 0) {
    // A node with non-zero value is a leaf node, add it to the map.
    output[node->name()] = value;
  } else {
    // A node with zero value might contain children.  If so we add them.
    for (auto child = node->first_node(); child;
         child = child->next_sibling()) {
      if (child->type() == rapidxml::node_type::node_element) {
        for (const auto &pair : flatten_xml_serialization(child, true)) {
          std::string joined_name = pair.first;
          if (prepend_parent) {
            joined_name = node->name() + std::string(".") + joined_name;
          }
          output[joined_name] = pair.second;
        }
      }
    }
  }
  return output;
}

template <typename FeatureType>
inline auto to_xml_buffer(const FeatureType &feature) {
  std::ostringstream oss;
  {
    cereal::XMLOutputArchive archive(oss);
    archive(feature);
  }
  std::istringstream fin(oss.str());

  std::vector<char> buffer((std::istreambuf_iterator<char>(fin)),
                           std::istreambuf_iterator<char>());
  buffer.push_back('\0');
  return buffer;
}

template <typename FeatureType>
inline std::map<std::string, std::string> to_map(const FeatureType &feature) {

  auto buffer = to_xml_buffer(feature);
  // Parse the buffer using the xml file parsing library into doc
  rapidxml::xml_document<> doc;
  doc.parse<0>(&buffer[0]);

  auto first_node = doc.first_node()->first_node();
  auto output = flatten_xml_serialization(first_node);
  return output;
}

// A special case for floating point features, otherwise they would
// be given a column name of "value0" by cereal.
inline std::map<std::string, std::string> to_map(const double &feature) {
  return {{"feature", std::to_string(feature)}};
}

template <typename FeatureType>
inline std::map<std::string, std::string>
to_map(const FeatureType &feature, double target, double target_variance,
       double prediction, double prediction_variance) {
  auto output = to_map(feature);
  output["target"] = std::to_string(target);
  output["target_variance"] = std::to_string(target_variance);
  output["prediction"] = std::to_string(prediction);
  output["prediction_variance"] = std::to_string(prediction_variance);
  return output;
}

/*
 * Extracts the i^th element from a dataset / prediction pair and
 * creates the corresponding column to value map.
 */
template <typename FeatureType, typename DistributionType>
inline std::map<std::string, std::string>
to_map(const RegressionDataset<FeatureType> &dataset,
       const DistributionBase<DistributionType> &predictions, std::size_t i) {
  assert(dataset.targets.size() == predictions.size());
  assert(i < dataset.features.size() && i >= 0);
  const auto ei = static_cast<Eigen::Index>(i);

  double target_variance = dataset.targets.get_diagonal(ei);
  double predict_variance = predictions.get_diagonal(ei);

  auto row = to_map(dataset.features[i], dataset.targets.mean[ei],
                    target_variance, predictions.mean[ei], predict_variance);

  row = map_join(row, dataset.metadata);
  row = map_join(row, predictions.metadata);

  return row;
}

/*
 * This helper function makes it easier to use range based
 * for loops that append a delimeter which can be followed
 * by this function to turn the last delimeter into a new line
 */
inline void replace_last_character_with_newline(std::ostream &stream) {
  stream.seekp(-1, std::ios_base::end);
  stream << std::endl;
}

template <typename FeatureType, typename DistributionType>
inline std::vector<std::string>
get_column_names(const RegressionDataset<FeatureType> &dataset,
                 const DistributionBase<DistributionType> &predictions) {
  std::set<std::string> keys;
  for (std::size_t i = 0; i < dataset.features.size(); ++i) {
    const auto next_keys = map_keys(to_map(dataset, predictions, i));
    keys.insert(next_keys.begin(), next_keys.end());
  }
  return std::vector<std::string>(keys.begin(), keys.end());
}

inline void write_row(std::ostream &stream,
                      const std::map<std::string, std::string> &row,
                      const std::vector<std::string> &columns) {
  for (const auto &col : columns) {
    if (map_contains(row, col)) {
      stream << row.at(col);
    }
    stream << ",";
  }
  replace_last_character_with_newline(stream);
}

inline void write_header(std::ostream &stream,
                         const std::vector<std::string> &columns) {
  for (const auto &col : columns) {
    stream << col << ",";
  }
  replace_last_character_with_newline(stream);
}

template <typename FeatureType, typename DistributionType>
inline void write_to_csv(std::ostream &stream,
                         const RegressionDataset<FeatureType> &dataset,
                         const DistributionBase<DistributionType> &predictions,
                         const std::vector<std::string> &columns) {

  for (std::size_t i = 0; i < dataset.features.size(); i++) {
    const auto row = to_map(dataset, predictions, i);
    write_row(stream, row, columns);
  }
}

template <typename FeatureType, typename CovarianceType>
inline void write_to_csv(std::ostream &stream,
                         const RegressionDataset<FeatureType> &dataset,
                         const DistributionBase<CovarianceType> &predictions,
                         bool include_header = true) {
  const auto columns = get_column_names(dataset, predictions);
  if (include_header) {
    write_header(stream, columns);
  }
  write_to_csv(stream, dataset, predictions, columns);
}

/*
 * Make it easier to write only a dataset (without predictions).
 */
template <typename FeatureType>
inline void write_to_csv(std::ostream &stream,
                         const RegressionDataset<FeatureType> &dataset,
                         bool include_header = true) {
  Eigen::VectorXd zeros = Eigen::VectorXd::Zero(dataset.targets.mean.size());
  MarginalDistribution zero_predictions(zeros);
  write_to_csv(stream, dataset, zero_predictions, include_header);
}

template <typename FeatureType, typename DistributionType,
          std::enable_if_t<is_distribution<DistributionType>::value, int> = 0>
inline void
write_to_csv(std::ostream &stream,
             const std::vector<RegressionDataset<FeatureType>> &datasets,
             const std::vector<DistributionType> &predictions) {
  const auto columns = get_column_names(datasets[0], predictions[0]);
  write_header(stream, columns);
  assert(datasets.size() == predictions.size());
  for (std::size_t i = 0; i < datasets.size(); i++) {
    write_to_csv(stream, datasets[i], predictions[i], columns);
  }
}

template <typename _Scalar, int _Rows, int _Cols>
inline void write_to_csv(std::ostream &stream,
                         const Eigen::Matrix<_Scalar, _Rows, _Cols> &x) {
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
      stream << x(i, j);
      if (j < x.cols() - 1) {
        stream << ",";
      }
    }
    if (i < x.rows() - 1) {
      stream << std::endl;
    }
  }
}
} // namespace albatross

#endif
