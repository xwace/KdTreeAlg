static bool compX(std::pair<Point, int> &pt1, std::pair<Point, int> &pt2)
{
    return pt1.first.x < pt2.first.x;
}  // compX

static bool compY(std::pair<Point, int> &pt1, std::pair<Point, int> &pt2)
{
    return pt1.first.y < pt2.first.y;
}  // compY

/*!
 * 建立 k-D Tree，采用vector保存k-D Tree的结构，每个子树的节点是该树对应区域的中间位置
 * @param points 待建立k-D Tree的点集
 * @param start 建立子树的起始点索引
 * @param end 建立子树的终止点索引
 * @param level 当前子树建立参考的维度，0表示参考x轴，,1表示参考y轴
 */
void makeTree(std::vector<std::pair<Point, int>> &points, int start, int end, int level)
{
    int length = end - start;
    // 当子树的节点数量小于2时，说明只有一个点了，不能再进行分割出子树了
    if (length < 2) return;
    // 当前子树的根节点的索引
    int mid = start + (length/2);

    if (level == 0)
    {
        // 参照x轴建立子树
        std::nth_element(points.begin()+start, points.begin()+mid, points.begin()+end, compX);
    }else
    {
        // 参照x轴建立子树
        std::nth_element(points.begin()+start, points.begin()+mid, points.begin()+end, compY);
    }

    // left 建立左边的子树
    makeTree(points, start, mid, (level+1)%2);
    // right 建立右边的子树
    makeTree(points, mid+1, end, (level+1)%2);
}


/**
  ******************************************************************************
  * @author         : oswin
  * @brief          : 二叉树的结构存放kd树,一个节点包含左右节点的指针
  ******************************************************************************
  */
//Points [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
struct KdNode {
    Point idx{};
    Ptr<KdNode> left{nullptr};
    Ptr<KdNode> right{nullptr};
    int sign;

    KdNode(Point idx_,Ptr<KdNode> left_,Ptr<KdNode> right_,int sign_):
    idx(idx_),left(std::move(left_)),right(std::move(right_)),sign(sign_){}
};

static bool compX1(Point* p1,Point* p2){
    return p1->x < p2->x;
}

bool compY1(const Point* p1, const Point* p2){
    return p1->y < p2->y;
}

Ptr<KdNode> build(vector<Point*> &points, int sign) {
    int len = (int) points.size();
    if (len == 0) {
        return nullptr;
    }

    int mid = len / 2;

    if (sign == 1) {
        std::nth_element(points.begin(), points.begin() + mid, points.end(), compX1);
    } else {
        std::nth_element(points.begin(), points.begin() + mid, points.end(), compY1);
    }

    //2 1 3 4 6 7 ->分割后: p1[2,1] 中值mid[3] p2[4,6,7]
    vector<Point *> p1(points.begin(), points.begin() + mid);
    vector<Point *> p2(points.begin() + mid + 1, points.end());//+1是为了不取中值

    sign = -sign;
    Ptr<KdNode> left = build(p1, sign);
    Ptr<KdNode> right = build(p2, sign);
    Ptr<KdNode> node = new KdNode(*points[mid], left, right, sign);

    return node;
}

void preorder(const Ptr<KdNode> &node) {
    cout << node->idx << endl;
    if (node->left)
        preorder(node->left);
    if (node->right)
        preorder(node->right);
}

void build_(vector<Point>& points){

    vector<Point*> pts;
    for (int i = 0; i < points.size(); ++i) {
        pts.emplace_back(&points[i]);
    }

    auto node = build(pts,1);
    preorder(node);
}

/*!
 * stack实现kd树创建
 * 注意对比stack和递归的联系:subtree的成员是递归函数的参数. 
 */

struct subTree {
    int nodeIdx{0};
    int first{-1};
    int last{-1};

    subTree(){}
    subTree(int first_, int last_,int nodeIdx_) : first(first_), last(last_), nodeIdx(nodeIdx_) {}
};

bool compX2(const Point& p1, const Point& p2){
    return p1.x < p2.x;
}

bool compY2(const Point& p1, const Point& p2){
    return p1.y < p2.y;
}

/**
  ******************************************************************************
  * @author         : oswin
  * @brief          : 查找最近邻点算法
  * @date           : 20230308
  ******************************************************************************
  */
void find_nearest(vector<Point> &points, const Point &pt, subTree &st, Point &nearest, int &min_dist) {
    if ((st.last - st.first) < 1) {
        return;
    }

    int mid = (st.first + st.last) / 2;

    //Should the new distance be shorter, update the nearest point
    auto dist = cv::norm(pt - points[mid]);
    if (dist < min_dist) {
        min_dist = dist;
        nearest = points[mid];
    }

    subTree subtree;
    subTree further_subtree;
    subTree left;
    subTree right;

    int delta = st.nodeIdx == 1 ? (pt.x - points[mid].x) : (pt.y - points[mid].y);
    left = subTree(st.first, mid, -st.nodeIdx);
    right = subTree(mid + 1, st.last, -st.nodeIdx);

    subtree = delta > 0 ? right : left;
    further_subtree = delta > 0 ? left : right;//兄弟子树

    find_nearest(points, pt, subtree, nearest, min_dist);

    //发现以目标点为圆心,最短距离为半径的圆包含mid父节点,需回溯继续搜索兄弟子树
    if (abs(delta) < min_dist) {
        find_nearest(points, pt, further_subtree, nearest, min_dist);
    }
}

/**
  ******************************************************************************
  * @author         : oswin
  * @brief          : 查找K近邻的算法. BBF做了回溯点丢弃的优化:
  *                   if(ns.size() == K && dist > ns.top().dist) continue;
  *                   inputs:points, outputs:ns
  ******************************************************************************
  */
struct PQueueElemMax {
    PQueueElemMax() : dist(0), point(0, 0) {}

    PQueueElemMax(float _dist, Point point_, subTree subtree_) : dist(_dist), point(point_), subtree(subtree_) {}

    friend bool operator<(const PQueueElemMax &left, const PQueueElemMax &right) {
        return left.dist < right.dist;//采用最大堆
    }

    float dist;
    Point point;
    subTree subtree;
};

struct PQueueElem {
    PQueueElem() : dist(0), point(0, 0) {}

    PQueueElem(float _dist, Point point_, subTree subtree_) : dist(_dist), point(point_), subtree(subtree_) {}

    friend bool operator<(const PQueueElem &left, const PQueueElem &right) {
        return left.dist > right.dist;//采用最小堆
    }

    float dist;
    Point point;
    subTree subtree;
};

//从根节点或者回溯节点搜索到叶子节点;期间,添加兄弟子树到回溯点优先队列pq;
void explore_to_leaf(vector<Point> &points, const Point &pt, subTree &st, int &thres_dist, int K,
                     priority_queue<PQueueElem> &pq,
                     priority_queue<PQueueElemMax> &ns) {

    if (st.last - st.first < 1) return;

    int mid = (st.first + st.last) / 2;

    auto dist = cv::norm(pt - points[mid]);

    if (ns.size() == K && dist > ns.top().dist) return;//裁剪掉本次回溯

    if (dist < thres_dist) {
        ns.emplace(dist, points[mid], subTree());
        if (ns.size() == K + 1)
            ns.pop();
    }

    subTree subtree;
    subTree further_subtree;
    subTree left;
    subTree right;

    int delta = st.nodeIdx == 1 ? (pt.x - points[mid].x) : (pt.y - points[mid].y);
    left = subTree(st.first, mid, -st.nodeIdx);
    right = subTree(mid + 1, st.last, -st.nodeIdx);

    subtree = delta > 0 ? right : left;
    further_subtree = delta > 0 ? left : right;//兄弟子树

    explore_to_leaf(points, pt, subtree, thres_dist, K, pq, ns);

    //dist比队列中最长距离短时入队
    if (ns.size() == K && dist > ns.top().dist) return;
    pq.emplace(dist, points[mid], further_subtree);
}

void find_nearest_neighbors(vector<Point> &points, const Point &pt, subTree &st, int &thres_dist, int K,
                            priority_queue<PQueueElemMax> &ns) {

    subTree initTree(0, (int) points.size(), 1);
    priority_queue<PQueueElem> pq;

    int mid = (initTree.first + initTree.last) / 2;
    auto dist = cv::norm(pt - points[mid]);
    pq.emplace(dist, points[mid], initTree);

    while (!pq.empty()) {

        //start one loop
        auto tree = pq.top().subtree;
        pq.pop();

        explore_to_leaf(points, pt, tree, thres_dist, K, pq, ns);
    }

}

/**
  ******************************************************************************
  * @author         : oswin
  * @brief          : 利用stack生成树,用vector存放,与前面的递归算法做对比
  ******************************************************************************
  */
void createTree(vector<Point> &points) {
    
    subTree initTree(0, (int) points.size(), 1);//attention: opencv used points.size()-1!!

    std::stack<subTree> stack;
    stack.emplace(initTree);

    while (!stack.empty()) {

        auto top = stack.top();
        stack.pop();

        int len = top.last - top.first;
        int idx = top.nodeIdx;

        if (len < 2) {
            continue;//len可以是0,1
        }

        int mid = len / 2 + top.first;

        if (idx == 1) {
            std::nth_element(points.begin() + top.first, points.begin() + mid, points.begin() + top.last,
                             compX2);
        } else {
            std::nth_element(points.begin() + top.first, points.begin() + mid, points.begin() + top.last,
                             compY2);
        }

        subTree left(top.first, mid, -idx);
        subTree right(mid + 1, top.last, -idx);

        stack.emplace(right);
        stack.emplace(left);
    }

    cout << "建kd树:\n " << points << endl;//输出结果

    /**
      ******************************************************************************
      * @author         : oswin
      * @brief          : 测试1--求最近邻点的接口
      ******************************************************************************
      */

    Point query_pt(2, 4), nearest{-1, -1};
    int min_dist{INT_MAX};
    find_nearest(points, query_pt, initTree, nearest, min_dist);
    cout << "返回最近邻点 nearest: " << nearest << endl;
    

    /**
      ******************************************************************************
      * @author         : oswin
      * @brief          : 测试2--K近邻调用接口.返回优先队列ns
      * @date           : K为自定义近邻系数
      ******************************************************************************
      */
    int thres_dist = 4;//自己设置最近邻的阈值, 阈值范围内的是近邻点
    priority_queue<PQueueElemMax> ns;//结果返回
    int K = 2;
    find_nearest_neighbors(points, query_pt, initTree, thres_dist, K, ns);
    cout << "nbs: size " << ns.top().point << endl;
    ns.pop();
    cout << "nbs: size " << ns.top().point << endl;
}









int main(){
    vector<Point> pts{{2, 3}, {5, 4}, {9, 6}, {4, 7}, {8, 1}, {7, 2}};
    createTree(pts);
//    build_(pts);

    /*vector<Point2f> points;
    for (int i = 0; i <pts.size() ; ++i) {
        points.emplace_back((float)pts[i].x,(float)pts[i].y);
    }
    KD::KDTree kd(points);*/

    /*std::vector<std::pair<Point, int>> points;
    for (int i = 0; i < pts.size(); ++i) {
        points.emplace_back(pts[i], i);
    }
    makeTree(points, 0, pts.size(), 0);*/
}
