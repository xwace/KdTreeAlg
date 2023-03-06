#include "kdTree.h"
using namespace cv;
using namespace std;

namespace KD{
    const int MAX_TREE_DEPTH = 32;

    KDTree::KDTree()
    {
        maxDepth = -1;
        normType = NORM_L2;
    }

    KDTree::KDTree(InputArray _points, bool _copyData)
    {
        maxDepth = -1;
        normType = NORM_L2;
        build(_points, _copyData);
    }

    KDTree::KDTree(InputArray _points, InputArray _labels, bool _copyData)
    {
        maxDepth = -1;
        normType = NORM_L2;
        build(_points, _labels, _copyData);
    }

    struct SubTree
    {
        SubTree() : first(0), last(0), nodeIdx(0), depth(0) {}
        SubTree(int _first, int _last, int _nodeIdx, int _depth)
                : first(_first), last(_last), nodeIdx(_nodeIdx), depth(_depth) {}
        int first;
        int last;
        int nodeIdx;
        int depth;
    };

    //std::nth_element排序的手写实现
    static float
    medianPartition( size_t* ofs, int a, int b, const float* vals )
    {
        int k, a0 = a, b0 = b;
        int middle = (a + b)/2;
        while( b > a )
        {
            int i0 = a, i1 = (a+b)/2, i2 = b;
            float v0 = vals[ofs[i0]], v1 = vals[ofs[i1]], v2 = vals[ofs[i2]];
            //ip为比较三个数后得到的中间数的下标
            int ip = v0 < v1 ? (v1 < v2 ? i1 : v0 < v2 ? i2 : i0) :
                     v0 < v2 ? (v1 == v0 ? i2 : i0): (v1 < v2 ? i2 : i1);
            float pivot = vals[ofs[ip]];
            //中间值放最后,用于后续所有元素排序;
            std::swap(ofs[ip], ofs[i2]);

            for( i1 = i0, i0--; i1 <= i2; i1++ )//从左到右遍历
                if( vals[ofs[i1]] <= pivot )
                {
                    i0++;
                    std::swap(ofs[i0], ofs[i1]);
                }
            if( i0 == middle )
                break;
            if( i0 > middle )
                b = i0 - (b == i0);
            else
                a = i0;
        }

        float pivot = vals[ofs[middle]];
        for( k = a0; k < middle; k++ )
        {
            CV_Assert(vals[ofs[k]] <= pivot);
        }
        for( k = b0; k > middle; k-- )
        {
            CV_Assert(vals[ofs[k]] >= pivot);
        }

        return vals[ofs[middle]];
    }

    static void
    computeSums( const Mat& points, const size_t* ofs, int a, int b, double* sums )
    {
        int i, j, dims = points.cols;
        const float* data = points.ptr<float>(0);
        for( j = 0; j < dims; j++ )
            sums[j*2] = sums[j*2+1] = 0;
        for( i = a; i <= b; i++ )
        {
            const float* row = data + ofs[i];
            for( j = 0; j < dims; j++ )
            {
                double t = row[j], s = sums[j*2] + t, s2 = sums[j*2+1] + t*t;
                sums[j*2] = s; sums[j*2+1] = s2;
            }
        }
    }

    void KDTree::build(InputArray _points, bool _copyData)
    {
        build(_points, noArray(), _copyData);
    }

    void KDTree::build(InputArray __points, InputArray __labels, bool _copyData)
    {
        Mat _points = __points.getMat(), _labels = __labels.getMat();
        _points = _points.reshape(1,_points.cols);

        CV_Assert(_points.type() == CV_32F && !_points.empty());
        std::vector<KDTree::Node>().swap(nodes);

        if( !_copyData )
            points = _points;
        else
        {
            points.release();
            points.create(_points.size(), _points.type());
        }

        int i, j, n = _points.rows, ptdims = _points.cols, top = 0;
        const float* data = _points.ptr<float>(0);
        float* dstdata = points.ptr<float>(0);
        size_t step = _points.step1();
        size_t dstep = points.step1();
        int ptpos = 0;
        labels.resize(n);
        const int* _labels_data = 0;

        if( !_labels.empty() )
        {
            int nlabels = _labels.checkVector(1, CV_32S, true);
            CV_Assert(nlabels == n);
            _labels_data = _labels.ptr<int>();
        }

        Mat sumstack(MAX_TREE_DEPTH*2, ptdims*2, CV_64F);
        SubTree stack[MAX_TREE_DEPTH*2];

        //存放点集的下标,排序后即为输出的结果
        std::vector<size_t> _ptofs(n);
        size_t* ptofs = &_ptofs[0];

        for( i = 0; i < n; i++ )
            ptofs[i] = i*step;

        nodes.push_back(Node());
        computeSums(points, ptofs, 0, n-1, sumstack.ptr<double>(top));
        stack[top++] = SubTree(0, n-1, 0, 0);
        int _maxDepth = 0;

        while( --top >= 0 )
        {
            int first = stack[top].first, last = stack[top].last;
            int depth = stack[top].depth, nidx = stack[top].nodeIdx;
            int count = last - first + 1, dim = -1;
            const double* sums = sumstack.ptr<double>(top);
            double invCount = 1./count, maxVar = -1.;

            if( count == 1 )
            {
                int idx0 = (int)(ptofs[first]/step);
                int idx = _copyData ? ptpos++ : idx0;
                nodes[nidx].idx = ~idx;
                if( _copyData )
                {
                    const float* src = data + ptofs[first];
                    float* dst = dstdata + idx*dstep;
                    for( j = 0; j < ptdims; j++ )
                        dst[j] = src[j];
                }
                labels[idx] = _labels_data ? _labels_data[idx0] : idx0;
                _maxDepth = std::max(_maxDepth, depth);
                continue;
            }

            // find the dimensionality with the biggest variance
            for( j = 0; j < ptdims; j++ )
            {
                double m = sums[j*2]*invCount;
                double varj = sums[j*2+1]*invCount - m*m;
                if( maxVar < varj )
                {
                    maxVar = varj;
                    dim = j;
                }
            }

            int left = (int)nodes.size(), right = left + 1;
            nodes.push_back(Node());
            nodes.push_back(Node());
            nodes[nidx].idx = dim;
            nodes[nidx].left = left;
            nodes[nidx].right = right;
            nodes[nidx].boundary = medianPartition(ptofs, first, last, data + dim);

            int middle = (first + last)/2;
            double *lsums = (double*)sums, *rsums = lsums + ptdims*2;
            computeSums(points, ptofs, middle+1, last, rsums);
            for( j = 0; j < ptdims*2; j++ )
                lsums[j] = sums[j] - rsums[j];
            cout<<"left right: "<<left<<" "<<right<<endl;
            stack[top++] = SubTree(first, middle, left, depth+1);
            stack[top++] = SubTree(middle+1, last, right, depth+1);
        }
        maxDepth = _maxDepth;

        vector<Point2f> vec = Mat_<Point2f>(_points);
        for (int k = 0; k < n; ++k) {
            cout<<"after being ordered: "<<vec[_ptofs[k]/2]<<endl;
        }

        for (auto node:nodes) {
            cout<<"tree node: "<<node.boundary<<" "<<node.left<<" "<<node.right<<endl;
        }
    }
}