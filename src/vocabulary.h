#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <vector>
#include <string>
#include <queue>
#include <boost/unordered_map.hpp>

namespace nplm
{

template <typename T>
struct compare_second
{
  bool operator()(const T &lhs, const T &rhs) const { return lhs.second < rhs.second; }
};

class vocabulary {
    std::vector<std::string> m_words;
    boost::unordered_map<std::string, int> m_index;
    int unk;
	bool unk_defined;

public:
    vocabulary() 
    { 
        //unk = insert_word("<unk>");
		unk_defined = 0;
    }

    vocabulary(const std::vector<std::string> &words)
      :
      m_words(words)
    {
        for (int i=0; i<words.size(); i++) {
            m_index[words[i]] = i;
			if (words[i] == "<unk>"){
				unk = m_index["<unk>"];
				unk_defined = 1;
			}
		}
		if (unk_defined == 0){
			std::cerr<<"Warning:: The vocabulary does not have unk! \n\
				Program will fail if the development set or test set have unknown words"<<std::endl;
		}
		//unk = m_index["<unk>"];
    }
	void build_vocab(const std::vector<std::string> &words) {
		//std::cerr<<"The vocab size is "<<words.size()<<std::endl;
		m_words = words;
        for (int i=0; i<words.size(); i++) {
            m_index[words[i]] = i;
			if (words[i] == "<unk>"){
				unk = m_index["<unk>"];
				unk_defined = 1;
			}			
		}
		if (unk_defined == 0){
			std::cerr<<"Warning:: The vocabulary does not have unk! \n\
				Program will fail if the development set or test set have unknown words"<<std::endl;
		}		
		//unk = m_index["<unk>"];		
	}
    int lookup_word(const std::string &word) const
	{
		boost::unordered_map<std::string, int>::const_iterator pos = m_index.find(word);
		if (pos != m_index.end())
			return pos->second;
		else
			return unk;
	}
	
	std::string get_word(const int id){
		//std::cerr<<"getting word for id "<<id<<std::endl;
		return m_words[id];
	}
    // lookup word using custom unknown-word id
    int lookup_word(const std::string &word, int unk) const
    {
        boost::unordered_map<std::string, int>::const_iterator pos = m_index.find(word);
		if (pos != m_index.end())
		    return pos->second;
		else if (!unk_defined) {
				std::cerr<<"Error:: Word "<<word<<" is not in the vocabulary and \n \
					vocabulary does not have <unk>"<<std::endl;
				std::exit(1);
		} else {
	    	return unk;
		}
    }

    int insert_word(const std::string &word)
	{
		int i = size();
		bool inserted = m_index.insert(make_pair(word, i)).second;
		if( inserted )
		{
			m_words.push_back(word);
		}
		return i;
	}

    int size() const { return m_words.size(); }

    // Inserts the most-frequent words from counts until vocab_size words are reached.
    // counts is a collection of pair<string,int>
    template <typename Map>
    int insert_most_frequent(const Map &counts, int vocab_size)
	{
		typedef std::pair<std::string,int> stringint;

		std::priority_queue<stringint,std::vector<stringint>,compare_second<stringint> > 
			q(compare_second<stringint>(), std::vector<stringint>(counts.begin(), counts.end()));

		int inserted = 0;
		insert_word("<unk>"); //infrequent words are all mapped to <unk>
		inserted++;
		unk = m_index["<unk>"];
		unk_defined = 1;
		while (size() < vocab_size && !q.empty())
		{
			insert_word(q.top().first);
			q.pop();
			inserted++;
		}
		return inserted;
	}

    const std::vector<std::string> &words() const { return m_words; }
	void print_vocabulary(){
		for (int i=0; i<m_words.size(); i++){
			std::cerr<<"Vocab item is "<<m_words.at(i)<<std::endl;
		}
	}
};

} // namespace nplm

#endif
