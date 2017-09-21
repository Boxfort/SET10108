#pragma once

#include <exception>
#include <stack>
#include <memory>
#include <mutex>

template<typename T>
class threadsafe_stack
{
private:
	std::stack<T> data;
	mutable std::mutex mut;
public:
	// Normal Constructor
	threadsafe_stack() { }
	// Copy constructor
	threadsafe_stack(const threadsafe_stack &other)
	{
		// We need to copy the data from the other stack. Lock other stack
		std::lock_guard<std::mutex> lock(other.mut);
		data = other.data;
	}

	void push(T value)
	{
		//Lock access to the object
		std::lock_guard<std::mutex> lock(mut);
		// Push value into internal stack
		data.push(value);
	}

	T pop()
	{
		// Lock access to object
		std::lock_guard<std::mutex> lock(mut);
		// Check if the stack is empty
		if (data.empty()) throw std::exception("Stack is empty");
		// Access the top of the stack
		auto res = data.top();
		// Remove the top item from stack
		data.pop();
		// Return resource
		return res;
	}

	bool empty() const
	{
		std::lock_guard<std::mutex> lock(mut);
		return data.empty();
	}
};