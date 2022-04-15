// g++ p.cpp -o p && ./p < 1.in
#include <iostream>
#include <string>
#include <algorithm>
#include <functional>
#include <cmath>
#include <array>
#include <vector>
#include <list>
#include <deque>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <unordered_map>
#include <bitset>
#include <climits>
#include <cfloat>

using namespace std;

#define SN scanf("\n")
#define SI(x) scanf("%d",&x)
#define SII(x,y) scanf("%d %d",&x,&y)
#define SL(x) scanf("%lld",&x)
#define SD(x) scanf("%lf",&x)
#define SC(x) scanf("%c",&x)
#define FOR(i, s, k) for(int i=s; i<k; i++)
#define REP(i, n) FOR(i, 0, n)
#define INF INT_MAX
#define EPS 1e-9
#define PI acos(-1)

typedef long long int lint;
typedef unsigned long long int ulint;
typedef pair<int, int> pii;
typedef pair<double, int> pdi;
typedef vector<int> vi;
typedef vector<lint> vl;
typedef vector<double> vd;
typedef vector<bool> vb;
typedef list<int> li;
typedef vector<vector<int>> vvi;
typedef vector<vector<double>> vvd;
typedef vector<pair<int, int>> vpii;
typedef list<pair<int, int>> lpii;
typedef vector<list<int>> vli;
typedef vector<list<pair<int, int>>> vlpii;
typedef vector<list<pair<double,int>>> vlpdi;


int main(){

    int t; SI(t); FOR(testcase, 1, t+1){
		
		double a; string b;
		SD(a); SN; cin >> b; cout<<a<<" "<<b<<endl;

		printf("Case #%d: \n", testcase);
    }

    return 0;
}

/* Union Find Integer */
map<int, pair<int, unsigned int>> Sets;
void AddSet(int x){ Sets.insert(make_pair(x, make_pair(x, 1))); }
int Find(int x){
	if(Sets[x].first == x){ return x; }
	else{ return Sets[x].first = Find(Sets[x].first); }
}
void Union(int x, int y) {
	int parentX = Find(x), parentY = Find(y);
	int rankX = Sets[parentX].second, rankY = Sets[parentY].second;
	if (parentX == parentY){ return; }
	else if(rankX < rankY){
		Sets[parentX].first = parentY;
		Sets[parentY].second += Sets[parentX].second;
	}else{
		Sets[parentY].first = parentX;
		Sets[parentX].second += Sets[parentY].second;
	}
}
int Size(int x){ return Sets[Find(x)].second; }
void Reset(){ Sets.clear(); }

/* Kruskal's Algorithm (Minimum spanning tree) */
typedef struct Edge{
	pii e;
	int w;
} Edge;
bool compareWeight(Edge a, Edge b){ return a.w < b.w; }
vector<Edge> Kruskal(vvi adj){
	int n = adj.size();
	vector<Edge> mst;
	vector<Edge> edges;
	FOR(i,1,n){
		FOR(j,0,i){
			edges.push_back({make_pair(j+1,i+1),adj[i][j]});
		}
	}
	sort(edges.begin(), edges.end(), compareWeight);
	REP(i,n){ AddSet(i); }
	for(auto e : edges){
		if(Find(e.e.first-1)!=Find(e.e.second-1)){
			mst.push_back(e);
			Union(e.e.first-1,e.e.second-1);
		}
	}
	Reset();
	return mst;
}

/* Prim's Algorithm (Minimum spanning tree) */
int Prim(vlpii adj){
	int n = adj.size();
	vb visited(n, false);
	priority_queue<pii, vpii, greater<pii>> Q;
	int cost = 0;
	Q.push(make_pair(0,1));

	while(!Q.empty()){
		pii p = Q.top(); Q.pop();
		int v = p.second;
		int w = p.first;

		if(!visited[v-1]){
			cost+=w;
			visited[v-1]=true;
			for(pii nei : adj[v-1]){
				if(!visited[nei.second-1]){
					Q.push(nei);
				}
			}
		}
	}
	return cost;
}

/* Dijkstra's Algorithm (Shortest paths from source) */
vi Dijkstra(vlpii adj, int src){
	int n = adj.size();
	priority_queue<pii, vpii, greater<pii>> PQ;
	vi dist(n,INF);
	vi parent(n, -1);
	dist[src-1] = 0;
	PQ.push(make_pair(0,src));
	while(!PQ.empty()){
		int u = PQ.top().second; PQ.pop();
		for(pii p : adj[u-1]){
			int v = p.second; int w = p.first;
			if(dist[u-1]+w<dist[v-1]){
				dist[v-1] = dist[u-1]+w;
				parent[v-1] = u;
				PQ.push(make_pair(dist[v-1],v));
			}
		}
	}
	return dist;
}

/* Bellman-Ford's Algorithm (Shortest paths from source and negative weight) */
pair<bool,vd> BellmanFordCycle(vlpdi adj, int src){
	int n = adj.size();
	deque<int> Q, Qp;
	vd dist(n,DBL_MAX);
	vi parent(n, -1);
	dist[src-1] = 0;
	Q.push_back(src);
	REP(i,n){
		
		while(!Q.empty()){
			int v;
			v = Q.front(); Q.pop_front();
			
			for(pdi p : adj[v-1]){
				int w = p.second; double c = p.first;
				if(dist[v-1]+c<dist[w-1]){
					dist[w-1] = dist[v-1]+c;
					parent[w-1] = v;
					if(find(Qp.begin(),Qp.end(),w)==Qp.end()){
						Qp.push_back(w);
					}
				}
			}
		}
		swap(Q,Qp);
	}
	return make_pair(!Q.empty(),dist);
}

/* Ford-Fulkerson's Algorithm (Maximum Flow) */
bool FordFulkersonBFS(vvi residualAdj, int s, int t, vi &parent){
	int n = residualAdj.size();
  	vb visited(n, false);
  	queue<int> q;
  	q.push(s);
  	visited[s-1] = true;
  	parent[s-1] = -1;

  	while(!q.empty()){
    	int u = q.front(); q.pop();
    	REP(v,n){
      		if(!visited[v] && residualAdj[u-1][v]>0) {
        		q.push(v+1);
        		parent[v] = u;
        		visited[v] = true;
      		}
    	}
  	}
  	return visited[t-1];
}
int FordFulkerson(vvi adj, int s, int t) {
	int n = adj.size();
	vvi residualAdj(n, vi(n));
	REP(i,n){ REP(j,n){ residualAdj[i][j] = adj[i][j]; } }
  	vi parent(n);
  	int maxFlow = 0;

  	while(FordFulkersonBFS(residualAdj, s, t, parent)){
    	int pathFlow = INF;
		int u;
		int v = t;
		while(v != s){
			u = parent[v-1];
			pathFlow = min(pathFlow, residualAdj[u-1][v-1]);
			v = u;
		}

		v = t;
		while(v != s){
			u = parent[v-1];
			residualAdj[u-1][v-1] -= pathFlow;
			residualAdj[v-1][u-1] += pathFlow;
			v = u;
		}

    	maxFlow += pathFlow;
  	}
  	return maxFlow;
}

/* Generate all permutations */
vi permutations(vi currentConfig){
    int m = currentConfig.size();

    REP(i,m){cout<<currentConfig[i];}cout<<endl;

    vi nextConfig = currentConfig;
    //Find largest k s.t. a[k]<a[k+1]
    int k=m-1-1; while(k>=0 && nextConfig[k]>=nextConfig[k+1]){ k--; }
    //If k does not exist then this is the last permutation
    if(k<0){ return vi(m,0); }
    //Find largest l s.t. a[k]<a[l]
    int l=m-1; while(l>=0 && currentConfig[k]>=currentConfig[l]){ l--; }
    //Swap value a[k] and a[l]
    swap(nextConfig[k],nextConfig[l]);
    //Reverse sequence from a[k+1] to a[m]
    reverse(nextConfig.begin()+k+1, nextConfig.end());

    permutations(nextConfig);

    return vi(m,0);
}

/* Compute GCD */
lint gcd(lint a, lint b){
	if(b==0){
		return a;
	}else{
		return gcd(b, a%b);
	}
}

/* Compite LCM */
lint lcm(lint a, lint b){
	return a*b/gcd(a,b);
}

/* Find Bezeout relation */
pair<lint,pair<lint,lint>> bezout(lint a, lint b){
	lint s = 0; lint sp = 1;
    lint t = 1; lint tp = 0;
    lint r = b; lint rp = a;
    lint q, temp;
    while(r!=0){
        q = rp/r;
        temp=rp; rp=r; r=temp-q*rp;
        temp=sp; sp=s; s=temp-q*sp;
        temp=tp; tp=t; t=temp-q*tp;
    }
    return make_pair(rp,make_pair(sp,tp));
}

/* Compute positive modulo */
lint modulo(lint a,lint b){ return (a%b+b)%b; }

/* Geometric structs */
typedef struct Point{
	double x;
	double y;
} Point;

/* Return the angle defined by ABC (degree) */
double angle(Point a, Point b, Point c){
    double aa = pow(b.x-a.x,2) + pow(b.y-a.y,2);
    double bb = pow(b.x-c.x,2) + pow(b.y-c.y,2);
    double cc = pow(c.x-a.x,2) + pow(c.y-a.y,2);
    return acos((aa+bb-cc)/sqrt(4*aa*bb))*180/PI;
}

/* Return the centroid of a polygon */
Point centroid(vector<Point> points){
    int n = points.size();
    Point centroid = {0, 0};
    double area = 0;
    double x0=0; double y0=0;
    double x1=0; double y1=0;
    double partialArea = 0.0;

    REP(i,n){
        x0 = points[i].x;
        y0 = points[i].y;
        x1 = points[(i+1)%n].x;
        y1 = points[(i+1)%n].y;
        partialArea = x0*y1 - x1*y0;
        area += partialArea;
        centroid.x += (x0+x1)*partialArea;
        centroid.y += (y0+y1)*partialArea;
    }

    area *= 0.5;
    centroid.x /= (6.0*area);
    centroid.y /= (6.0*area);

    return centroid;
}