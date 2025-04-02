/**
 * Job Matching Chatbot
 * Core functionality for the AI-powered chatbot in the job matching system
 */

class Chatbot {
  constructor(options) {
    this.userType = options.userType; // 'job_seeker' or 'employer'
    this.containerId = options.containerId || 'chatbot-container';
    this.triggerId = options.triggerId || 'chatbot-trigger';
    this.userData = options.userData || {};
    this.isMinimized = true;
    this.isInitialized = false;
    this.context = options.context || [];
    this.mockDelay = options.mockDelay || { min: 500, max: 2000 };
    this.suggestions = options.suggestions || [];
    
    this.initialize();
  }
  
  initialize() {
    if (this.isInitialized) return;
    
    this.createChatbotDOM();
    this.addEventListeners();
    this.isInitialized = true;
    
    // Add welcome message after a slight delay
    setTimeout(() => this.addBotMessage(this.getWelcomeMessage()), 500);
  }
  
  createChatbotDOM() {
    // Create chatbot trigger button
    const triggerBtn = document.createElement('div');
    triggerBtn.id = this.triggerId;
    triggerBtn.className = 'chatbot-trigger';
    triggerBtn.innerHTML = '<i data-lucide="message-circle"></i>';
    document.body.appendChild(triggerBtn);
    
    // Create chatbot container
    const chatbotContainer = document.createElement('div');
    chatbotContainer.id = this.containerId;
    chatbotContainer.className = 'chatbot-container minimized';
    
    // Create chatbot header
    const chatbotHeader = document.createElement('div');
    chatbotHeader.className = 'chatbot-header';
    chatbotHeader.innerHTML = `
      <h3>AI Assistant</h3>
      <div class="chatbot-header-actions">
        <button class="minimize-btn"><i data-lucide="minus"></i></button>
        <button class="close-btn"><i data-lucide="x"></i></button>
      </div>
    `;
    
    // Create chatbot body
    const chatbotBody = document.createElement('div');
    chatbotBody.className = 'chatbot-body';
    
    // Add suggestions if provided
    if (this.suggestions.length > 0) {
      const suggestionsDiv = document.createElement('div');
      suggestionsDiv.className = 'suggestions';
      
      this.suggestions.forEach(suggestion => {
        const chip = document.createElement('div');
        chip.className = 'suggestion-chip';
        chip.textContent = suggestion;
        chip.addEventListener('click', () => this.handleUserInput(suggestion));
        suggestionsDiv.appendChild(chip);
      });
      
      chatbotBody.appendChild(suggestionsDiv);
    }
    
    // Create chatbot footer with input
    const chatbotFooter = document.createElement('div');
    chatbotFooter.className = 'chatbot-footer';
    chatbotFooter.innerHTML = `
      <input type="text" class="chatbot-input" placeholder="Type your question...">
      <button class="send-button"><i data-lucide="send"></i></button>
    `;
    
    // Assemble the chatbot
    chatbotContainer.appendChild(chatbotHeader);
    chatbotContainer.appendChild(chatbotBody);
    chatbotContainer.appendChild(chatbotFooter);
    
    // Add to document
    document.body.appendChild(chatbotContainer);
    
    // Initialize Lucide icons
    if (window.lucide) {
      lucide.createIcons();
    }
  }
  
  addEventListeners() {
    // Trigger button
    const triggerBtn = document.getElementById(this.triggerId);
    triggerBtn.addEventListener('click', () => this.toggleChatbot(true));
    
    // Minimize button
    const minimizeBtn = document.querySelector(`#${this.containerId} .minimize-btn`);
    minimizeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleChatbot(false);
    });
    
    // Close button
    const closeBtn = document.querySelector(`#${this.containerId} .close-btn`);
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.toggleChatbot(false);
    });
    
    // Header click to toggle
    const header = document.querySelector(`#${this.containerId} .chatbot-header`);
    header.addEventListener('click', () => {
      if (this.isMinimized) {
        this.toggleChatbot(true);
      }
    });
    
    // Send button
    const sendBtn = document.querySelector(`#${this.containerId} .send-button`);
    sendBtn.addEventListener('click', () => this.sendMessage());
    
    // Input keypress (Enter)
    const inputField = document.querySelector(`#${this.containerId} .chatbot-input`);
    inputField.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.sendMessage();
      }
    });
  }
  
  toggleChatbot(show) {
    const container = document.getElementById(this.containerId);
    const trigger = document.getElementById(this.triggerId);
    
    if (show) {
      container.classList.remove('minimized');
      trigger.classList.add('hidden');
      this.isMinimized = false;
      
      // Focus input field
      const inputField = document.querySelector(`#${this.containerId} .chatbot-input`);
      inputField.focus();
    } else {
      container.classList.add('minimized');
      trigger.classList.remove('hidden');
      this.isMinimized = true;
    }
  }
  
  sendMessage() {
    const inputField = document.querySelector(`#${this.containerId} .chatbot-input`);
    const message = inputField.value.trim();
    
    if (message) {
      this.addUserMessage(message);
      this.handleUserInput(message);
      inputField.value = '';
    }
  }
  
  addUserMessage(message) {
    const chatBody = document.querySelector(`#${this.containerId} .chatbot-body`);
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user-message';
    messageDiv.innerHTML = `<div class="message-content">${message}</div>`;
    
    chatBody.appendChild(messageDiv);
    this.scrollToBottom();
  }
  
  addBotMessage(message) {
    const chatBody = document.querySelector(`#${this.containerId} .chatbot-body`);
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message bot-message';
    messageDiv.innerHTML = `<div class="message-content">${message}</div>`;
    
    chatBody.appendChild(messageDiv);
    this.scrollToBottom();
  }
  
  addLoadingIndicator() {
    const chatBody = document.querySelector(`#${this.containerId} .chatbot-body`);
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message bot-message loading-message';
    loadingDiv.innerHTML = `
      <div class="message-content">
        <div class="loading-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    
    chatBody.appendChild(loadingDiv);
    this.scrollToBottom();
    
    return loadingDiv;
  }
  
  removeLoadingIndicator() {
    const loadingMessage = document.querySelector(`#${this.containerId} .loading-message`);
    if (loadingMessage) {
      loadingMessage.remove();
    }
  }
  
  scrollToBottom() {
    const chatBody = document.querySelector(`#${this.containerId} .chatbot-body`);
    chatBody.scrollTop = chatBody.scrollHeight;
  }
  
  getWelcomeMessage() {
    if (this.userType === 'job_seeker') {
      return `Hi there! I'm your AI career assistant. I can help you find the best job matches, filter opportunities, and provide personalized recommendations. What would you like to know?`;
    } else {
      return `Welcome! I'm your recruiting assistant. I can help you find top candidates, analyze applicant qualifications, and provide insights on your job postings. How can I assist you today?`;
    }
  }
  
  getMockResponseDelay() {
    return Math.floor(Math.random() * (this.mockDelay.max - this.mockDelay.min + 1)) + this.mockDelay.min;
  }
  
  /**
   * Handle user input and generate appropriate responses
   * In a real implementation, this would call an API
   */
  handleUserInput(message) {
    // Show loading indicator
    const loadingIndicator = this.addLoadingIndicator();
    
    // Process the message after a simulated delay
    setTimeout(() => {
      this.removeLoadingIndicator();
      
      // Add to context for simulated "memory"
      this.context.push({ role: 'user', content: message });
      
      // Generate response based on user type and message content
      const response = this.generateResponse(message);
      
      // Add bot response to chat
      this.addBotMessage(response.message);
      
      // If response includes a data visualization, add it
      if (response.html) {
        this.addBotMessage(response.html);
      }
      
      // Add to context
      this.context.push({ role: 'assistant', content: response.message });
    }, this.getMockResponseDelay());
  }
  
  /**
   * Generate a response based on user type and message
   * This is a simple mock implementation - in production this would call an LLM API
   */
  generateResponse(message) {
    const lowerMsg = message.toLowerCase();
    
    // Shared responses for both user types
    if (lowerMsg.includes('hello') || lowerMsg.includes('hi ') || lowerMsg.includes('hey')) {
      return { message: `Hello! How can I help you today?` };
    }
    
    if (lowerMsg.includes('thank')) {
      return { message: `You're welcome! Is there anything else I can help you with?` };
    }
    
    // User type specific responses
    if (this.userType === 'job_seeker') {
      return this.generateJobSeekerResponse(lowerMsg);
    } else {
      return this.generateEmployerResponse(lowerMsg);
    }
  }
  
  /**
   * Generate job seeker specific responses
   */
  generateJobSeekerResponse(message) {
    // Top job matches
    if (message.includes('top job') || message.includes('best match') || message.includes('highest match')) {
      return {
        message: `Here are your top job matches based on your skills and preferences:`,
        html: this.generateJobMatchesHTML()
      };
    }
    
    // Recent jobs
    if (message.includes('recent') || message.includes('newest') || message.includes('latest')) {
      return {
        message: `These are the most recently posted jobs that match your profile:`,
        html: this.generateRecentJobsHTML()
      };
    }
    
    // Skill recommendations
    if (message.includes('skill') || message.includes('improve') || message.includes('learn') || message.includes('recommendation')) {
      return {
        message: `Based on your profile and job market trends, I recommend developing these skills:`,
        html: this.generateSkillRecommendationsHTML()
      };
    }
    
    // Salary information
    if (message.includes('salary') || message.includes('pay') || message.includes('compensation')) {
      return {
        message: `Here's salary information for your top matching roles:`,
        html: this.generateSalaryInfoHTML()
      };
    }
    
    // Default response
    return {
      message: `I can help you find job matches, discover recent openings, get skill recommendations, or check salary information. What would you like to know?`
    };
  }
  
  /**
   * Generate employer specific responses
   */
  generateEmployerResponse(message) {
    // Top candidates
    if (message.includes('top candidate') || message.includes('best candidate') || message.includes('highest match')) {
      return {
        message: `Here are the top candidates for your job posting:`,
        html: this.generateTopCandidatesHTML()
      };
    }
    
    // Skill gap analysis
    if (message.includes('skill gap') || message.includes('skill analysis') || message.includes('missing skill')) {
      return {
        message: `Based on your applicant pool, here's a skill gap analysis:`,
        html: this.generateSkillGapHTML()
      };
    }
    
    // Application statistics
    if (message.includes('stat') || message.includes('application') || message.includes('dashboard')) {
      return {
        message: `Here are your current application statistics:`,
        html: this.generateApplicationStatsHTML()
      };
    }
    
    // Optimize job posting
    if (message.includes('optimize') || message.includes('improve') || message.includes('better') || message.includes('posting')) {
      return {
        message: `Here are recommendations to optimize your job posting for better matches:`,
        html: this.generateJobOptimizationHTML()
      };
    }
    
    // Default response
    return {
      message: `I can help you find top candidates, analyze skill gaps, view application statistics, or optimize your job postings. What would you like to know?`
    };
  }
  
  // JOB SEEKER HTML GENERATORS
  
  generateJobMatchesHTML() {
    return `
      <div class="results-card">
        <h4>Software Engineer at TechCorp</h4>
        <div class="results-card-content">
          <div><span class="match-score">98% Match</span> · $120,000 - $150,000</div>
          <div>San Francisco, CA · Posted 2 days ago</div>
          <div class="skills-list">
            <span class="skill-tag">Python</span>
            <span class="skill-tag">React</span>
            <span class="skill-tag">AWS</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>Senior Developer at InnoSoft</h4>
        <div class="results-card-content">
          <div><span class="match-score">92% Match</span> · $110,000 - $140,000</div>
          <div>Remote · Posted 5 days ago</div>
          <div class="skills-list">
            <span class="skill-tag">JavaScript</span>
            <span class="skill-tag">Node.js</span>
            <span class="skill-tag">MongoDB</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>Full Stack Engineer at GrowthStart</h4>
        <div class="results-card-content">
          <div><span class="match-score">87% Match</span> · $100,000 - $130,000</div>
          <div>New York, NY · Posted 1 week ago</div>
          <div class="skills-list">
            <span class="skill-tag">TypeScript</span>
            <span class="skill-tag">Django</span>
            <span class="skill-tag">PostgreSQL</span>
          </div>
        </div>
      </div>
    `;
  }
  
  generateRecentJobsHTML() {
    return `
      <div class="results-card">
        <h4>Backend Developer at DataFlow</h4>
        <div class="results-card-content">
          <div><span class="match-score">85% Match</span> · $115,000 - $135,000</div>
          <div>Seattle, WA · Posted today</div>
          <div class="skills-list">
            <span class="skill-tag">Go</span>
            <span class="skill-tag">Kubernetes</span>
            <span class="skill-tag">MySQL</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>ML Engineer at AILabs</h4>
        <div class="results-card-content">
          <div><span class="match-score">79% Match</span> · $125,000 - $155,000</div>
          <div>Boston, MA · Posted yesterday</div>
          <div class="skills-list">
            <span class="skill-tag">Python</span>
            <span class="skill-tag">TensorFlow</span>
            <span class="skill-tag">NLP</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>Frontend Developer at UXPro</h4>
        <div class="results-card-content">
          <div><span class="match-score">82% Match</span> · $95,000 - $120,000</div>
          <div>Remote · Posted 2 days ago</div>
          <div class="skills-list">
            <span class="skill-tag">React</span>
            <span class="skill-tag">TypeScript</span>
            <span class="skill-tag">CSS</span>
          </div>
        </div>
      </div>
    `;
  }
  
  generateSkillRecommendationsHTML() {
    return `
      <div class="results-card">
        <h4>Recommended Skills to Develop</h4>
        <div class="results-card-content">
          <div class="skills-list">
            <span class="skill-tag">Kubernetes</span>
            <span class="skill-tag">GraphQL</span>
            <span class="skill-tag">Swift</span>
            <span class="skill-tag">React Native</span>
            <span class="skill-tag">System Design</span>
          </div>
          <p>These skills appear frequently in job postings that match your profile but are not yet in your skill set.</p>
        </div>
      </div>
    `;
  }
  
  generateSalaryInfoHTML() {
    return `
      <div class="results-card">
        <h4>Salary Information for Your Matches</h4>
        <div class="results-card-content">
          <p><b>Software Engineer:</b> $110,000 - $145,000</p>
          <p><b>Senior Developer:</b> $130,000 - $165,000</p>
          <p><b>Full Stack Engineer:</b> $105,000 - $140,000</p>
          <p><b>Machine Learning Engineer:</b> $125,000 - $170,000</p>
          <p>Based on your experience level and location preferences.</p>
        </div>
      </div>
    `;
  }
  
  // EMPLOYER HTML GENERATORS
  
  generateTopCandidatesHTML() {
    return `
      <div class="results-card">
        <h4>Alex Johnson - Software Engineer</h4>
        <div class="results-card-content">
          <div><span class="match-score">96% Match</span> · 5 years experience</div>
          <div>San Francisco, CA · Available immediately</div>
          <div class="skills-list">
            <span class="skill-tag">Python</span>
            <span class="skill-tag">React</span>
            <span class="skill-tag">AWS</span>
            <span class="skill-tag">Node.js</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>Morgan Smith - Senior Developer</h4>
        <div class="results-card-content">
          <div><span class="match-score">94% Match</span> · 7 years experience</div>
          <div>Remote · 2 week notice</div>
          <div class="skills-list">
            <span class="skill-tag">Java</span>
            <span class="skill-tag">Spring</span>
            <span class="skill-tag">Kubernetes</span>
          </div>
        </div>
      </div>
      
      <div class="results-card">
        <h4>Taylor Williams - Full Stack Engineer</h4>
        <div class="results-card-content">
          <div><span class="match-score">90% Match</span> · 4 years experience</div>
          <div>New York, NY · Available in 3 weeks</div>
          <div class="skills-list">
            <span class="skill-tag">JavaScript</span>
            <span class="skill-tag">React</span>
            <span class="skill-tag">Express.js</span>
            <span class="skill-tag">MongoDB</span>
          </div>
        </div>
      </div>
    `;
  }
  
  generateSkillGapHTML() {
    return `
      <div class="results-card">
        <h4>Skill Gap Analysis</h4>
        <div class="results-card-content">
          <p><b>Most common skills in applicant pool:</b></p>
          <div class="skills-list">
            <span class="skill-tag">JavaScript (87%)</span>
            <span class="skill-tag">React (72%)</span>
            <span class="skill-tag">Python (58%)</span>
            <span class="skill-tag">AWS (45%)</span>
          </div>
          
          <p><b>Underrepresented skills in applicant pool:</b></p>
          <div class="skills-list">
            <span class="skill-tag">Kubernetes (12%)</span>
            <span class="skill-tag">GraphQL (18%)</span>
            <span class="skill-tag">Go (22%)</span>
          </div>
          
          <p>Consider modifying your job posting to attract candidates with these underrepresented skills.</p>
        </div>
      </div>
    `;
  }
  
  generateApplicationStatsHTML() {
    return `
      <div class="results-card">
        <h4>Application Statistics</h4>
        <div class="results-card-content">
          <p><b>Total Applications:</b> 247</p>
          <p><b>Qualified Candidates:</b> 68 (27.5%)</p>
          <p><b>High Match Score (>90%):</b> 23 (9.3%)</p>
          <p><b>Average Years of Experience:</b> 4.2 years</p>
          <p><b>Application Source:</b> LinkedIn (42%), Indeed (28%), Direct (30%)</p>
        </div>
      </div>
    `;
  }
  
  generateJobOptimizationHTML() {
    return `
      <div class="results-card">
        <h4>Job Posting Optimization</h4>
        <div class="results-card-content">
          <p><b>Recommended improvements:</b></p>
          <ul>
            <li>Add more specific technical requirements to attract specialized candidates</li>
            <li>Highlight remote work options more prominently</li>
            <li>Include salary range to attract more qualified applicants</li>
            <li>Mention career growth opportunities and mentorship</li>
            <li>Simplify the skills section to focus on must-haves vs. nice-to-haves</li>
          </ul>
          <p>These changes could increase your qualified candidate pool by an estimated 35%.</p>
        </div>
      </div>
    `;
  }
}

// Expose to window
window.Chatbot = Chatbot; 