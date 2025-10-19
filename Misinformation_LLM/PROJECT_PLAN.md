# CMPE297 Fact-Checking System
## Project Requirements and Action Items

### Immediate Action Items (Meeting 5)
- [ ] All module contracts pushed to GitHub (input/output formats, dependencies)
  - Danny: Claim Extraction contract
  - Adam: Vector DB contract
  - Sam: Fact Checking contract (DONE)
  - Akshay: LLM Reasoning contract (PENDING)
  - Yuxiao: UI Interface contract
  - Stephen: Integration & Testing contract (DONE)
- [ ] API interfaces defined between modules
  - Claim Extraction → Vector DB query format
  - Vector DB → Fact Checking evidence format
  - Fact Checking → LLM input format
  - LLM → UI display format
- [ ] Module workloads defined
  - Sam: Claim type parsing, feature extraction, validation eval
  - Adam: Chunking strategy, embedding model selection
  - Akshay: Prompt engineering, OpenAI integration
  - Others: TBD
- [ ] Dataset selection and acquisition
  - Choose from: FEVER, SciFact, PUBHEALTH, AVeriTeC, etc.
  - Minimum 50-100 claims for validation
  - Define ground truth format
- [ ] User story examples (start-to-finish flow)

### Technical Specifications Needed
- [ ] JSON format standards
  - Claim structure (from Danny)
  - Evidence format (from Adam)
  - Fact-check output (from Sam)
- [ ] Qdrant configuration
  - Chunking strategy for long documents
  - Embedding model selection (Hugging Face)
  - Relevance score interpretation (cosine similarity/distance)
- [ ] OpenAI API integration
  - Model selection
  - Prompt templates
  - Error handling

### Future Work
- [ ] CI/CD pipeline setup
- [ ] Performance optimization
- [ ] Scalability testing
- [ ] Additional dataset integration
- [ ] Multi-language support
- [ ] API rate limiting
- [ ] Caching strategy
- [ ] Logging and monitoring