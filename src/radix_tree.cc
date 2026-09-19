#include "mini_llama/radix_tree.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace mini_llama {

// 根节点本身不保存 token，仅作为所有压缩边的统一起点。
RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::Clear() {
  // 清空根节点的所有子树；unique_ptr 会递归释放整棵树。
  root_.children.clear();
  root_.terminal = false;
  root_.key.clear();
  terminal_count_ = 0;
}

bool RadixTree::empty() const { return terminal_count_ == 0; }

size_t RadixTree::size() const { return terminal_count_; }

// 计算待插入/匹配序列与压缩边之间的最长公共前缀长度。
size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b) {
  size_t i = 0;
  while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
    ++i;
  }
  return i;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
  // 空序列由根节点的 terminal 标记表示，不能创建空 key 子节点。
  if (tokens.empty()) {
    if (!root_.terminal) {
      root_.terminal = true;
      ++terminal_count_;
    }
    return;
  }
  InsertInto(&root_, tokens);
}

void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
  // 每条边保存一段连续 token，而不是单个 token，从而压缩只有一个子分支的路径。
  auto child_it = parent->children.find(suffix[0]);
  if (child_it == parent->children.end()) {
    // 没有共享首 token，直接挂载一条新的终止边。
    auto child = std::make_unique<Node>();
    child->key = std::move(suffix);
    child->terminal = true;
    parent->children[child->key[0]] = std::move(child);
    ++terminal_count_;
    return;
  }

  std::unique_ptr<Node>& child = child_it->second;
  size_t common = CommonPrefixLength(suffix, 0, child->key);

  if (common == child->key.size()) {
    if (common == suffix.size()) {
      // 已完整覆盖现有边：重复插入不增加终止序列计数。
      if (!child->terminal) {
        child->terminal = true;
        ++terminal_count_;
      }
      return;
    }

    std::vector<int> remaining(suffix.begin() + static_cast<long>(common),
                               suffix.end());
    // 现有压缩边完全匹配，但新序列还有剩余，递归插入子节点。
    InsertInto(child.get(), std::move(remaining));
    return;
  }

  auto split = std::make_unique<Node>();
  // 只匹配了部分边时，将公共前缀提升为中间节点。
  split->key.assign(child->key.begin(),
                    child->key.begin() + static_cast<long>(common));

  child->key.erase(child->key.begin(),
                   child->key.begin() + static_cast<long>(common));
  int old_child_first = child->key[0];
  split->children[old_child_first] = std::move(child);

  if (common == suffix.size()) {
    // 新序列正好结束于公共前缀，中间节点本身成为终止节点。
    split->terminal = true;
    ++terminal_count_;
  } else {
    // 公共前缀之后仍有新 token，创建新分支并标记其末端。
    auto new_child = std::make_unique<Node>();
    new_child->key.assign(suffix.begin() + static_cast<long>(common), suffix.end());
    new_child->terminal = true;
    split->children[new_child->key[0]] = std::move(new_child);
    ++terminal_count_;
  }

  child_it->second = std::move(split);
}

size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
  // 返回输入序列从位置 0 开始、能够沿树完整匹配的 token 数量。
  if (tokens.empty()) {
    return 0;
  }

  const Node* node = &root_;
  size_t pos = 0;
  while (pos < tokens.size()) {
    auto child_it = node->children.find(tokens[pos]);
    if (child_it == node->children.end()) {
      // 当前 token 没有对应分支，匹配在此处停止。
      return pos;
    }

    const Node* child = child_it->second.get();
    size_t common = CommonPrefixLength(tokens, pos, child->key);
    pos += common;
    if (common < child->key.size()) {
      // 输入在压缩边内部发生分歧，只能返回已经匹配的部分。
      return pos;
    }
    node = child;
  }
  return pos;
}

}  // namespace mini_llama
