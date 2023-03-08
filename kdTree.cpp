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
    
    
    
    
     struct PQueueElem
    {
        PQueueElem() : dist(0), idx(0) {}
        PQueueElem(float _dist, int _idx) : dist(_dist), idx(_idx) {}
        float dist;
        int idx;
    };

    int KDTree::findNearest(InputArray _vec, int K, int emax,
                            OutputArray _neighborsIdx, OutputArray _neighbors,
                            OutputArray _dist, OutputArray _labels) const

    {
        Mat vecmat = _vec.getMat();
        CV_Assert( vecmat.isContinuous() && vecmat.type() == CV_32F && vecmat.total() == (size_t)points.cols );
        const float* vec = vecmat.ptr<float>();
        K = std::min(K, points.rows);
        int ptdims = points.cols;

        CV_Assert(K > 0 && (normType == NORM_L2 || normType == NORM_L1));

        AutoBuffer<uchar> _buf((K+1)*(sizeof(float) + sizeof(int)));
        int* idx = (int*)_buf.data();
        float* dist = (float*)(idx + K + 1);
        //ncount统计找到的近邻点数目
        int i, j, ncount = 0, e = 0;

        int qsize = 0, maxqsize = 1 << 10;
        AutoBuffer<uchar> _pqueue(maxqsize*sizeof(PQueueElem));
        PQueueElem* pqueue = (PQueueElem*)_pqueue.data();

        //emax代表搜索叶子数;是一个经验值,限制搜索次数,可避免一直回溯到根节点
        emax = std::max(emax, 1);

        for( e = 0; e < emax; )
        {
            float d, alt_d = 0.f;
            //nidx代表子树的序号
            int nidx;

            if( e == 0 )
                nidx = 0;
            else
            {
                // take the next node from the priority queue
                if( qsize == 0 )
                    break;

                //nidx代表子树的序号
                //弹出回溯点优先队列最小距离的点
                nidx = pqueue[0].idx;
                alt_d = pqueue[0].dist;

                //优先队列更新top位(2->3),pro:2,3,10,6->post:3,6,10,2
                if( --qsize > 0 )
                {
                    std::swap(pqueue[0], pqueue[qsize]);
                    d = pqueue[0].dist;
                    for( i = 0;;)
                    {
                        int left = i*2 + 1, right = i*2 + 2;
                        if( left >= qsize )
                            break;
                        if( right < qsize && pqueue[right].dist < pqueue[left].dist )
                            left = right;
                        if( pqueue[left].dist >= d )
                            break;
                        std::swap(pqueue[i], pqueue[left]);
                        i = left;
                    }
                }

                // [回溯点的距离]> max[K近邻点集的距离],跳过该次回溯
                if( ncount == K && alt_d > dist[ncount-1] )
                    continue;
            }

            for(;;)
            {
                if( nidx < 0 )//叶子节点跳出循环;结束一次搜索
                    break;

                //根节点或者回溯点开始搜索,直到遇到叶子节点为一次完整的搜索
                const Node& n = nodes[nidx];

                //取到叶子节点
                if( n.idx < 0 )
                {
                    i = ~n.idx;//叶子节点对应点的真实索引;负号为了区分叶子节点与其他节点
                    const float* row = points.ptr<float>(i);
                    if( normType == NORM_L2 )
                        for( j = 0, d = 0.f; j < ptdims; j++ )
                        {
                            float t = vec[j] - row[j];
                            d += t*t;
                        }
                    else
                        for( j = 0, d = 0.f; j < ptdims; j++ )
                            d += std::abs(vec[j] - row[j]);

                    dist[ncount] = d;//!!!output 近邻点距离
                    idx[ncount] = i;//!!!output 近邻点索引

                    //升序排序 (dist0,idx0) < (dist1,idx1)
                    for( i = ncount-1; i >= 0; i-- )
                    {
                        if( dist[i] <= d )
                            break;
                        //上浮最小值 pair<idx,dist>
                        std::swap(dist[i], dist[i+1]);
                        std::swap(idx[i], idx[i+1]);
                    }
                    ncount += ncount < K;//k近邻还没到找足够数量,继续进队列
                    e++;
                    break;
                }

                //下面这段代码决定搜索方向;nidx避免了重新计算下一节点的位置
                int alt;
                if( vec[n.idx] <= n.boundary )//vec是目标点
                {
                    nidx = n.left;
                    alt = n.right;
                }
                else
                {
                    nidx = n.right;
                    alt = n.left;
                }

                d = vec[n.idx] - n.boundary;
                if( normType == NORM_L2 )
                    d = d*d + alt_d;
                else
                    d = std::abs(d) + alt_d;

                // subtree prunning 实现剪枝
                //已找到K近邻的前提,距离大的回溯节点跳过
                if( ncount == K && d > dist[ncount-1] )//dist是存放近邻点集的距离
                    continue;

                // add alternative subtree to the priority queue
                //添加兄弟子树到优先队列;升序
                pqueue[qsize] = PQueueElem(d, alt);
                for( i = qsize; i > 0; )
                {
                    int parent = (i-1)/2;
                    if( parent < 0 || pqueue[parent].dist <= d )
                        break;
                    std::swap(pqueue[i], pqueue[parent]);
                    i = parent;
                }
                qsize += qsize+1 < maxqsize;
            }
        }

        Mat neibours = Mat(K, 1, CV_32S, idx);
        cout<<"neighbours: \n"<<neibours<<endl;

        K = std::min(K, ncount);
        if( _neighborsIdx.needed() )
        {
            _neighborsIdx.create(K, 1, CV_32S, -1, true);
            Mat nidx = _neighborsIdx.getMat();
            Mat(nidx.size(), CV_32S, &idx[0]).copyTo(nidx);
        }
        if( _dist.needed() )
            sqrt(Mat(K, 1, CV_32F, dist), _dist);

        if( _neighbors.needed() || _labels.needed() )
            getPoints(Mat(K, 1, CV_32S, idx), _neighbors, _labels);
        return K;
    }

    void KDTree::getPoints(InputArray _idx, OutputArray _pts, OutputArray _labels) const
    {
        Mat idxmat = _idx.getMat(), pts, labelsmat;
        CV_Assert( idxmat.isContinuous() && idxmat.type() == CV_32S &&
                   (idxmat.cols == 1 || idxmat.rows == 1) );
        const int* idx = idxmat.ptr<int>();
        int* dstlabels = 0;

        int ptdims = points.cols;
        int i, nidx = (int)idxmat.total();
        if( nidx == 0 )
        {
            _pts.release();
            _labels.release();
            return;
        }

        if( _pts.needed() )
        {
            _pts.create( nidx, ptdims, points.type());
            pts = _pts.getMat();
        }

        if(_labels.needed())
        {
            _labels.create(nidx, 1, CV_32S, -1, true);
            labelsmat = _labels.getMat();
            CV_Assert( labelsmat.isContinuous() );
            dstlabels = labelsmat.ptr<int>();
        }
        const int* srclabels = !labels.empty() ? &labels[0] : 0;

        for( i = 0; i < nidx; i++ )
        {
            int k = idx[i];
            CV_Assert( (unsigned)k < (unsigned)points.rows );
            const float* src = points.ptr<float>(k);
            if( !pts.empty() )
                std::copy(src, src + ptdims, pts.ptr<float>(i));
            if( dstlabels )
                dstlabels[i] = srclabels ? srclabels[k] : k;
        }
    }


    const float* KDTree::getPoint(int ptidx, int* label) const
    {
        CV_Assert( (unsigned)ptidx < (unsigned)points.rows);
        if(label)
            *label = labels[ptidx];
        return points.ptr<float>(ptidx);
    }
}
