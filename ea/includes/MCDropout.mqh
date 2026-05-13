#ifndef MCDROPOUT_MQH
#define MCDROPOUT_MQH

#include "Defines.mqh"
#include "OnnxAgent.mqh"

//=============================================================================
// CMCDropout — thin wrapper that calls OnnxAgent.InferMC().
// Kept for naming consistency with mk3. In mk4, MC inference is fully
// delegated to the _mc.onnx model via repeated OnnxRun calls inside OnnxAgent.
//=============================================================================

class CMCDropout
{
private:
   COnnxAgent* m_agent;

public:
   CMCDropout() : m_agent(NULL) {}

   void SetAgent(COnnxAgent* agent) { m_agent = agent; }

   //--- Run T stochastic passes → mu (probability of LONG), sigma_ep
   void Infer(const float &x[], float &mu, float &sigma_ep)
   {
      if(m_agent == NULL || !m_agent.IsLoaded())
      {
         mu = 0.5f; sigma_ep = (float)MC_UNCERTAINTY_CAP + 0.01f;
         return;
      }
      m_agent.InferMC(x, MC_T, mu, sigma_ep);
   }

   //--- Quick check: is uncertainty above the cap?
   bool IsUncertain(float sigma_ep) const
   {
      return sigma_ep > MC_UNCERTAINTY_CAP;
   }

   //--- Direction from probability: +1, 0, -1
   int Direction(float mu) const
   {
      if(mu > CONF_THRESHOLD)  return  1;
      if(mu < 1.0f - (float)CONF_THRESHOLD) return -1;
      return 0;
   }
};

#endif // MCDROPOUT_MQH
