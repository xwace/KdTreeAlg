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
 * stack实现kd树
 */
struct subTree {
    int nodeIdx{0};
    int first{-1};
    int last{-1};

    subTree(int first_, int last_,int nodeIdx_) : first(first_), last(last_), nodeIdx(nodeIdx_) {}
};

bool compX2(const Point& p1, const Point& p2){
    return p1.x < p2.x;
}

bool compY2(const Point& p1, const Point& p2){
    return p1.y < p2.y;
}

void createTree(vector<Point> &points) {
    subTree initTree(0, (int)points.size(),1);//attention: opencv used points.size()-1!!

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

    cout<<points<<endl;//输出结果
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
