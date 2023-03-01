#include <string>
#include <vector>

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/stringpiece.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/util/command_line_flags.h>
#include <tensorflow/core/framework/tensor_slice.h>

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::pair;
using tf::Tensor;


struct Prediction
{
	int label;
	float score;
	float y;
	float x;
	float h;
	float w;
};


struct ImageReader
{
	static vector<Tensor> read(const string& path)
	{
		tf::Scope scope = tf::Scope::NewRootScope();
		auto image_raw = ops::ReadFile(scope.WithOpName("file_reader"), path);
		auto image_decoded = ops::DecodeImage(scope.WithOpName("file_decoder"), image_raw);
		auto image_uint8 = ops::Cast(scope.WithOpName("uint8_caster"), image_decoded, tf::DT_UINT8);
		auto image_expanded = ops::ExpandDims(scope.WithOpName("expand_dims"), image_uint8, 0);

		tf::GraphDef graph;
		auto status = scope.ToGraphDef(&graph);
		if (!status.ok()) {
			cout << "ImageRead: graph failed" << endl;
			exit(1);
		}

		tf::ClientSession session(scope);
		vector<Tensor> image;
		status = session.Run({image_expanded}, &image);
		if(!status.ok()) {
			cout << "ImageRead: run session failed" << endl;
			exit(1);
		}

		return image;
	}
};

class Model
{
public:
	Model(const string& model_path)
	{
		load(model_path);
	}

	void load(const string& model_path)
	{
		tf::SessionOptions session_options;
		session_options.config.mutable_gpu_options()->set_allow_growth(true);

		auto status = tf::LoadSavedModel(
			session_options,
			tf::RunOptions(),
			model_path,
			{"serve"},
			&bundle);

		if (!status.ok()) {
			cout << "Wupsi dupsi model couldn't be loaded." << endl;
			exit(1);
		}
	}

	vector<Prediction> operator()(const string& img_path) const
	{
		return predict(img_path);
	}

	vector<Prediction> predict(const string& img_path) const
	{
		auto img = ImageReader::read(img_path);

		// configure inputs and outputs
		const string input_node = "serving_default_input_tensor:0";
		vector<pair<string, Tensor>> input_config = {{input_node, img[0]}};
		vector<string> output_config = {
			"StatefulPartitionedCall:0",  //detection_anchor_indices
			"StatefulPartitionedCall:1",  //detection_boxes
			"StatefulPartitionedCall:2",  //detection_classes
			"StatefulPartitionedCall:3",  //detection_multiclass_scores
			"StatefulPartitionedCall:4",  //detection_scores
			"StatefulPartitionedCall:5"   //num_detections
		};

		// run prediction
		vector<Tensor> raw_preds;
		tf::Session* session = bundle.GetSession();
		auto status = session->Run(input_config, output_config, {}, &raw_preds);
		if (!status.ok()) {
			cout << "Predict: run prediction failed" << endl;
			exit(1);
		}
		
		// convert to Prediction
		auto bboxes = raw_preds[1].tensor<float, 3>();
		auto labels = raw_preds[2].tensor<float, 2>();
		auto scores = raw_preds[4].tensor<float, 2>();

		vector<Prediction> preds;
		preds.reserve(labels.size());

		for (int i = 0; i < labels.size(); ++i)
		{
			preds.push_back(
				Prediction{
					static_cast<int>(labels(0, i)),
					scores(0, i),
					bboxes(0, i, 0),  // y
					bboxes(0, i, 1),  // x
					bboxes(0, i, 2),  // h
					bboxes(0, i, 3),  // w
				}
			);
		}

		return preds;
	}

private:
	tf::SavedModelBundle bundle;
};
