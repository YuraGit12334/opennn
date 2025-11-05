//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../opennn/dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;
using namespace std;

namespace kmeans
{
    using namespace opennn;

    vector<Index> k_means(const Tensor<type, 2>& data,
                          const Index K,
                          const unsigned int max_iterations = 100,
                          const unsigned int seed = 12345)
    {
        const Index samples = data.dimension(0);
        const Index features = data.dimension(1);

        if (samples == 0 || features == 0 || K == 0)
        {
            return {};
        }

        std::mt19937 rng(seed);
        std::uniform_int_distribution<Index> dist(0, samples - 1);

        Tensor<type, 2> centroids(K, features);
        for (Index k = 0; k < K; ++k)
        {
            const Index random_index = dist(rng);
            for (Index f = 0; f < features; ++f)
            {
                centroids(k, f) = data(random_index, f);
            }
        }

        vector<Index> labels(samples, 0);
        vector<Index> previous_labels(samples, Index(-1));

        for (unsigned int iteration = 0; iteration < max_iterations; ++iteration)
        {
            for (Index i = 0; i < samples; ++i)
            {
                type best_distance = std::numeric_limits<type>::max();
                Index best_k = 0;

                for (Index k = 0; k < K; ++k)
                {
                    type distance = 0;
                    for (Index f = 0; f < features; ++f)
                    {
                        const type diff = data(i, f) - centroids(k, f);
                        distance += diff * diff;
                    }

                    if (distance < best_distance)
                    {
                        best_distance = distance;
                        best_k = k;
                    }
                }

                labels[i] = best_k;
            }

            if (labels == previous_labels)
            {
                break;
            }

            previous_labels = labels;

            Tensor<type, 2> new_centroids(K, features);
            new_centroids.setZero();

            vector<Index> counts(K, 0);

            for (Index i = 0; i < samples; ++i)
            {
                const Index label = labels[i];
                ++counts[label];
                for (Index f = 0; f < features; ++f)
                {
                    new_centroids(label, f) += data(i, f);
                }
            }

            for (Index k = 0; k < K; ++k)
            {
                if (counts[k] == 0)
                {
                    const Index random_index = dist(rng);
                    for (Index f = 0; f < features; ++f)
                    {
                        new_centroids(k, f) = data(random_index, f);
                    }
                    continue;
                }

                for (Index f = 0; f < features; ++f)
                {
                    new_centroids(k, f) /= static_cast<type>(counts[k]);
                }
            }

            centroids = new_centroids;
        }

        return labels;
    }

    vector<unique_ptr<Dataset>> split_dataset_by_clusters(const Dataset& original,
                                                          const vector<Index>& labels,
                                                          const Index K,
                                                          const type training_ratio = type(0.75),
                                                          const type selection_ratio = type(0.15),
                                                          const type testing_ratio = type(0.10))
    {
        vector<unique_ptr<Dataset>> cluster_datasets;
        cluster_datasets.reserve(static_cast<size_t>(K));

        const vector<Index> used_variable_indices = original.get_used_variable_indices();
        const vector<Dataset::RawVariable> raw_variables = original.get_raw_variables();
        const dimensions input_dimensions = original.get_input_dimensions();
        const dimensions target_dimensions = original.get_target_dimensions();

        for (Index k = 0; k < K; ++k)
        {
            vector<Index> indices;
            for (Index i = 0; i < static_cast<Index>(labels.size()); ++i)
            {
                if (labels[i] == k)
                {
                    indices.push_back(i);
                }
            }

            if (indices.empty())
            {
                continue;
            }

            Tensor<type, 2> cluster_data = original.get_data_from_indices(indices, used_variable_indices);

            auto cluster_dataset = std::make_unique<Dataset>(static_cast<Index>(indices.size()),
                                                             input_dimensions,
                                                             target_dimensions);

            cluster_dataset->set_display(false);
            cluster_dataset->set_raw_variables(raw_variables);
            cluster_dataset->set_dimensions("Input", input_dimensions);
            cluster_dataset->set_dimensions("Target", target_dimensions);
            cluster_dataset->set_data(cluster_data);
            cluster_dataset->set_sample_uses("Training");
            cluster_dataset->split_samples_random(training_ratio, selection_ratio, testing_ratio);

            cluster_datasets.push_back(std::move(cluster_dataset));
        }

        return cluster_datasets;
    }
}

namespace
{
    struct AccuracyMetrics
    {
        double accuracy = std::numeric_limits<double>::quiet_NaN();
        double loss = std::numeric_limits<double>::quiet_NaN();
    };

    AccuracyMetrics evaluate_subset(NeuralNetwork& network,
                                    const Dataset& dataset,
                                    const string& sample_use)
    {
        const Index samples = dataset.get_samples_number(sample_use);
        if (samples == 0)
        {
            return {};
        }

        const Tensor<type, 2> inputs = dataset.get_data(sample_use, "Input");
        const Tensor<type, 2> targets = dataset.get_data(sample_use, "Target");
        const Tensor<type, 2> outputs = network.calculate_outputs<2, 2>(inputs);

        const Index outputs_number = network.get_outputs_number();

        Index correct = 0;
        type cumulative_loss = 0;

        for (Index sample = 0; sample < samples; ++sample)
        {
            Index predicted_label = 0;
            Index target_label = 0;

            if (outputs_number == 1)
            {
                const type output_value = outputs(sample, 0);
                const type target_value = targets(sample, 0);

                predicted_label = output_value >= type(0.5) ? 1 : 0;
                target_label = target_value >= type(0.5) ? 1 : 0;

                const type clipped_output = std::clamp(output_value, type(1.0e-7), type(1.0 - 1.0e-7));
                cumulative_loss -= target_value * std::log(clipped_output)
                    + (type(1) - target_value) * std::log(type(1) - clipped_output);
            }
            else
            {
                type max_output = outputs(sample, 0);
                for (Index output_index = 1; output_index < outputs_number; ++output_index)
                {
                    if (outputs(sample, output_index) > max_output)
                    {
                        max_output = outputs(sample, output_index);
                        predicted_label = output_index;
                    }
                }

                type max_target = targets(sample, 0);
                for (Index output_index = 1; output_index < outputs_number; ++output_index)
                {
                    if (targets(sample, output_index) > max_target)
                    {
                        max_target = targets(sample, output_index);
                        target_label = output_index;
                    }
                }

                const type clipped_output = std::clamp(outputs(sample, target_label), type(1.0e-7), type(1.0 - 1.0e-7));
                cumulative_loss -= std::log(clipped_output);
            }

            if (predicted_label == target_label)
            {
                ++correct;
            }
        }

        AccuracyMetrics metrics;
        metrics.accuracy = static_cast<double>(correct) / static_cast<double>(samples);
        metrics.loss = static_cast<double>(cumulative_loss) / static_cast<double>(samples);
        return metrics;
    }

    struct ClusterTrainingSummary
    {
        Index cluster_index = 0;
        Index samples = 0;
        double elapsed_seconds = 0.0;
        string stopping_condition;
        type final_training_error = std::numeric_limits<type>::quiet_NaN();
        type final_selection_error = std::numeric_limits<type>::quiet_NaN();
        AccuracyMetrics training_metrics;
        AccuracyMetrics selection_metrics;
        AccuracyMetrics testing_metrics;
        Tensor<Index, 2> confusion;
    };

    ClusterTrainingSummary train_cluster_network(Index cluster_index,
                                                 unique_ptr<Dataset> cluster_dataset,
                                                 const dimensions& input_dimensions,
                                                 const dimensions& complexity_dimensions,
                                                 const dimensions& output_dimensions)
    {
        ClusterTrainingSummary summary;
        summary.cluster_index = cluster_index;
        summary.samples = cluster_dataset->get_samples_number();

        ClassificationNetwork network(input_dimensions, complexity_dimensions, output_dimensions);
        network.set_parameters_glorot();
        network.set_display(false);

        TrainingStrategy training_strategy(&network, cluster_dataset.get());
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        if (auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm()))
        {
            adam->set_learning_rate(type(5.0e-4));
            adam->set_beta_1(type(0.9));
            adam->set_beta_2(type(0.999));
            adam->set_batch_size(64);
            adam->set_maximum_epochs_number(250);
            adam->set_display_period(10);
            adam->set_loss_goal(type(1.0e-4));
        }

        const auto training_start = chrono::steady_clock::now();
        const TrainingResults training_results = training_strategy.train();
        const auto training_end = chrono::steady_clock::now();

        summary.elapsed_seconds = chrono::duration_cast<chrono::duration<double>>(training_end - training_start).count();
        summary.stopping_condition = training_results.write_stopping_condition();

        const auto extract_error = [](const Tensor<type, 1>& errors) -> type
        {
            if (errors.size() == 0)
            {
                return std::numeric_limits<type>::quiet_NaN();
            }
            return errors(errors.size() - 1);
        };

        summary.final_training_error = extract_error(training_results.training_error_history);
        summary.final_selection_error = extract_error(training_results.selection_error_history);

        summary.training_metrics = evaluate_subset(network, *cluster_dataset, "Training");
        summary.selection_metrics = evaluate_subset(network, *cluster_dataset, "Selection");
        summary.testing_metrics = evaluate_subset(network, *cluster_dataset, "Testing");

        TestingAnalysis testing_analysis(&network, cluster_dataset.get());
        summary.confusion = testing_analysis.calculate_confusion();

        return summary;
    }
}

int main()
{
    try
    {
        cout << "OpenNN. Amazon bag-of-words MLP example." << endl;

        const string dataset_file = "../data/amazon_cells_labelled_data.txt";
        Dataset dataset(dataset_file, ";", true, false);
        dataset.set_display(false);

        const Index raw_variables_number = dataset.get_raw_variables_number();
        if (raw_variables_number < 2)
        {
            throw runtime_error("Dataset must contain at least one input column and one target column.");
        }

        dataset.set_variable_uses("Input");
        const Index target_raw_index = raw_variables_number - 1;
        dataset.set_raw_variable_use(target_raw_index, "Target");

        const Index targets_number = dataset.get_variables_number("Target");
        if (targets_number == 1)
        {
            dataset.set_raw_variable_type(target_raw_index, Dataset::RawVariableType::Binary);
        }

        const dimensions input_dimensions = dataset.get_input_dimensions();
        const dimensions output_dimensions = dataset.get_target_dimensions();
        const dimensions complexity_dimensions = {256, 128, 64};

        const Index clusters = 5;
        cout << "Running k-means clustering into " << clusters << " clusters..." << endl;

        const Tensor<type, 2> input_data = dataset.get_data_variables("Input");
        const vector<Index> labels = kmeans::k_means(input_data, clusters, 100, 12345);

        if (labels.empty())
        {
            throw runtime_error("Unable to assign clusters to samples.");
        }

        cout << "Splitting dataset by clusters and preparing parallel trainings..." << endl;
        auto cluster_datasets = kmeans::split_dataset_by_clusters(dataset, labels, clusters);

        if (cluster_datasets.empty())
        {
            throw runtime_error("No clusters were generated for training.");
        }

        vector<future<ClusterTrainingSummary>> futures;
        futures.reserve(cluster_datasets.size());

        for (Index cluster_index = 0; cluster_index < static_cast<Index>(cluster_datasets.size()); ++cluster_index)
        {
            auto cluster_dataset = std::move(cluster_datasets[cluster_index]);
            futures.emplace_back(async(launch::async,
                                       [cluster_index,
                                        dataset_ptr = std::move(cluster_dataset),
                                        input_dimensions,
                                        complexity_dimensions,
                                        output_dimensions]() mutable
                                       {
                                           return train_cluster_network(cluster_index,
                                                                        std::move(dataset_ptr),
                                                                        input_dimensions,
                                                                        complexity_dimensions,
                                                                        output_dimensions);
                                       }));
        }

        vector<ClusterTrainingSummary> summaries;
        summaries.reserve(futures.size());

        for (auto& future_summary : futures)
        {
            summaries.push_back(future_summary.get());
        }

        cout << fixed << setprecision(4);
        cout << "\nParallel training finished. Cluster summaries:\n";

        for (const ClusterTrainingSummary& summary : summaries)
        {
            cout << "\nCluster " << summary.cluster_index
                 << " | samples: " << summary.samples
                 << " | time: " << summary.elapsed_seconds << " s" << endl;
            cout << "  Stopping condition: " << summary.stopping_condition << endl;

            if (!std::isnan(summary.final_training_error))
            {
                cout << "  Final training error: " << summary.final_training_error << endl;
            }
            else
            {
                cout << "  Final training error: n/a" << endl;
            }

            if (!std::isnan(summary.final_selection_error))
            {
                cout << "  Final selection error: " << summary.final_selection_error << endl;
            }
            else
            {
                cout << "  Final selection error: n/a" << endl;
            }

            const auto report_subset = [&](const string& subset, const AccuracyMetrics& metrics)
            {
                if (std::isnan(metrics.accuracy))
                {
                    cout << "  " << subset << " accuracy: n/a (no samples)" << endl;
                    return;
                }

                cout << "  " << subset << " accuracy: " << setprecision(2)
                     << metrics.accuracy * 100.0 << " %";
                cout << setprecision(4) << " | loss: " << metrics.loss << endl;
            };

            report_subset("Training", summary.training_metrics);
            report_subset("Selection", summary.selection_metrics);
            report_subset("Testing", summary.testing_metrics);

            if (summary.confusion.size() != 0)
            {
                cout << "  Confusion matrix:\n" << summary.confusion << endl;
            }
            else
            {
                cout << "  Confusion matrix: n/a" << endl;
            }
        }

        cout << "\nAll clusters trained successfully!" << endl;
        cout << "Good bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA)
